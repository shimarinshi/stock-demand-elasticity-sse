import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings
import os
from tqdm import tqdm
import osqp
from scipy import sparse
import gc

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==========================================
# 全局配置与参数
# ==========================================
MARKETTYPE_SSE = 1  # 上证A股代码
FACTOR_MARKET_ID = 'P9701'  # 上证A股因子市场编码
FACTOR_PORTFOLIO = 1  # 目标投资组合类型
ROLLING_BETA_WINDOW = 60  # 经典因子模型使用5年(60个月)的月度滚动窗口
GAMMA = 1  # 风险厌恶系数
OPTIMIZATION_TOL = 1e-10
MAX_ALLOWED_ELASTICITY=2000#仅删除极少值！

# --------------------------
# 模型配置
# --------------------------
MODELS_CONFIG = {
    "FF6": ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
}
# 全量因子列名映射
ALL_FACTOR_COL_MAPPING = {'RiskPremium2': 'MKT', 'SMB2': 'SMB', 'HML2': 'HML', 'RMW2': 'RMW', 'CMA2': 'CMA'}


# ==========================================
# 基础工具函数
# ==========================================
def format_permno(series):
    """统一将股票代码格式化为6位补零字符串"""
    return series.astype(str).str.split('.').str[0].str.zfill(6)


def standardize_monthly(series, group_col, df):
    """月度截面标准化"""
    scaler = StandardScaler()

    def _standardize(group):
        group_clean = group.replace([np.inf, -np.inf], np.nan).dropna()
        if len(group_clean) < 5 or group_clean.std() < 1e-8:
            return group - group.mean()
        scaler.fit(group_clean.values.reshape(-1, 1))
        return pd.Series(scaler.transform(group.values.reshape(-1, 1)).flatten(), index=group.index)

    return df.groupby(group_col)[series].transform(_standardize)


def load_data(file_path, name="数据文件"):
    """通用文件读取函数"""
    if not os.path.exists(file_path):
        print(f"警告: 找不到 {name} ({file_path})")
        return pd.DataFrame()
    for enc in ['gbk', 'utf-8', 'gb2312']:
        try:
            df = pd.read_csv(file_path, encoding=enc, na_values=['inf', '-inf', 'NaN'])
            df.columns = df.columns.str.strip()
            df = df.replace([np.inf, -np.inf], np.nan)
            if 'market_equity' in df.columns:
                df['market_equity'] = pd.to_numeric(df['market_equity'], errors='coerce')
            return df
        except Exception:
            continue
    print(f"错误: 无法读取 {name} ({file_path})")
    return pd.DataFrame()


# ==========================================
# 核心1：加载月度数据与因子数据（新增行业标签合并）
# ==========================================
def load_and_clean_data(monthly_data_path, factor_data_path, monthly_data_dir='.'):
    """
    加载并清洗数据，不剔除市值后20%，并合并行业标签
    """
    print("=" * 60)
    print("步骤1：加载上证A股月度数据与月度因子数据")
    print("=" * 60)
    # 1. 加载月度收益率数据及财务数据
    monthly_df = load_data(monthly_data_path, "月度收益率数据")
    if 'yearmonth' in monthly_df.columns and 'year_month' not in monthly_df.columns:
        monthly_df = monthly_df.rename(columns={'yearmonth': 'year_month'})
    leverage_df = load_data(os.path.join(monthly_data_dir, 'monthly_leverage.csv'), "杠杆率数据")
    roa_investment_df = load_data(os.path.join(monthly_data_dir, 'monthly_roa_investment.csv'), "ROA数据")
    asset_growth_df = load_data(os.path.join(monthly_data_dir, 'monthly_asset_growth.csv'), "资产增长数据")
    if not leverage_df.empty:
        leverage_df = leverage_df.rename(
            columns={'yearmonth': 'year_month'}) if 'yearmonth' in leverage_df.columns else leverage_df
        monthly_df = pd.merge(monthly_df, leverage_df[['permno', 'year_month', 'monthly_leverage']],
                              on=['permno', 'year_month'], how='left')
    if not roa_investment_df.empty:
        roa_investment_df = roa_investment_df.rename(
            columns={'yearmonth': 'year_month'}) if 'yearmonth' in roa_investment_df.columns else roa_investment_df
        monthly_df = pd.merge(monthly_df,
                              roa_investment_df[['permno', 'year_month', 'monthly_roa', 'monthly_invest_return']],
                              on=['permno', 'year_month'], how='left')
    if not asset_growth_df.empty:
        asset_growth_df = asset_growth_df.rename(
            columns={'yearmonth': 'year_month'}) if 'yearmonth' in asset_growth_df.columns else asset_growth_df
        monthly_df = pd.merge(monthly_df, asset_growth_df[['permno', 'year_month', 'asset_growth']],
                              on=['permno', 'year_month'], how='left')

    # 过滤上证A股
    monthly_df = monthly_df[monthly_df['Markettype'] == MARKETTYPE_SSE].reset_index(drop=True)
    monthly_df['permno'] = format_permno(monthly_df['permno'])

    # 【修改点1】不剔除市值后20%，仅做基础清洗
    monthly_df = monthly_df.dropna(subset=['market_equity', 'exret'])

    sse_permnos = monthly_df['permno'].unique()

    # ================= 【新增：加载并合并行业标签】 =================
    industry_lookup = load_data('industry_num.csv', name="行业分类数据")
    if industry_lookup.empty:
        raise ValueError("未找到 industry_num.csv 文件")

    industry_lookup['permno'] = format_permno(industry_lookup['Stkcd'])
    industry_lookup['ind_label'] = industry_lookup['Nnindcd'].str[0].str.upper()

    # 合并到主表
    monthly_df = pd.merge(monthly_df, industry_lookup[['permno', 'ind_label']], on='permno', how='left')
    # ==========================================================

    print(f"上证A股月度样本筛选完成（全市场，未剔除市值），共 {len(sse_permnos)} 只股票，{len(monthly_df)} 条月度记录")

    # 2. 加载月度因子数据
    factor_df = load_data(factor_data_path, "月度因子数据")
    if 'TradingMonth' in factor_df.columns:
        factor_df['year_month'] = pd.to_datetime(factor_df['TradingMonth'], errors='coerce')
        factor_df['year_month'] = (factor_df['year_month'].dt.year * 100 + factor_df['year_month'].dt.month).fillna(
            0).astype(int)
    factor_df = factor_df[
        (factor_df['MarkettypeID'] == FACTOR_MARKET_ID) &
        (factor_df['Portfolios'] == FACTOR_PORTFOLIO)
        ].reset_index(drop=True)
    factor_df = factor_df.rename(columns={k: v for k, v in ALL_FACTOR_COL_MAPPING.items() if k in factor_df.columns})
    factor_df = factor_df[['year_month'] + list(ALL_FACTOR_COL_MAPPING.values())].dropna().sort_values(
        'year_month').reset_index(drop=True)
    mom_df = load_data('momfactor.csv', "月度动量因子数据")
    if not mom_df.empty:
        if 'TradingMonth' in mom_df.columns:
            mom_df['year_month'] = pd.to_datetime(mom_df['TradingMonth'], errors='coerce')
            mom_df['year_month'] = (mom_df['year_month'].dt.year * 100 + mom_df['year_month'].dt.month).fillna(
                0).astype(int)
        mom_df = mom_df[
            (mom_df['MarkettypeID'] == FACTOR_MARKET_ID) &
            (mom_df['Quantile'] == '10%') &
            (mom_df['StockClass'] == 0) &
            (mom_df['FormationPeriod'] == 12)
            ].reset_index(drop=True)
        mom_df = mom_df.rename(columns={'MomRe2': 'MOM'})
        mom_df = mom_df[['year_month', 'MOM']].dropna()
        factor_df = pd.merge(factor_df, mom_df, on='year_month', how='inner')
        print(f"动量因子数据加载完成，有效时间范围：{mom_df['year_month'].min()} 至 {mom_df['year_month'].max()}")
    else:
        raise ValueError("动量因子文件momfactor.csv加载失败，请检查文件路径与格式")

    print(f"全量因子数据（含动量）加载完成，时间范围：{factor_df['year_month'].min()} 至 {factor_df['year_month'].max()}")
    return monthly_df, factor_df, sse_permnos


# ==========================================
# 核心2：【修改点2】分行业计算价格传递系数 PPT
# ==========================================
def run_fama_macbeth_by_industry(df):
    """
    按行业(ind_label)分组，每个行业单独跑Fama-MacBeth回归计算PPT
    """
    print("\n" + "=" * 60)
    print("步骤2：分行业Fama-MacBeth回归 计算行业专属PPT(φ)")
    print("=" * 60)

    data = df.copy()
    data = data[(data['prc'] > 0) & (data['vol'] > 0) & (data['market_equity'] > 0)]

    # 基础特征构建
    data['log_prc'] = np.log(data['prc'].abs())
    data['log_prc_lag'] = data.groupby('permno')['log_prc'].shift(1)
    data['log_vol'] = np.log(data['vol'])
    data['log_me'] = np.log(data['market_equity'])
    data = data.replace([np.inf, -np.inf], np.nan)

    data['std_log_vol'] = standardize_monthly('log_vol', 'year_month', data)
    data['std_log_me'] = standardize_monthly('log_me', 'year_month', data)

    base_vars = ['log_prc', 'log_prc_lag', 'std_log_vol', 'std_log_me', 'monthly_roa', 'monthly_invest_return']
    vars_model = [v for v in base_vars if v in data.columns]
    for col in vars_model:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 回归标的：下一期超额收益率
    data['exret_next'] = data.groupby('permno')['exret'].shift(-1)
    data = data.dropna(subset=['exret_next', 'ind_label'] + vars_model)

    # 按行业分组循环
    industry_list = data['ind_label'].dropna().unique()
    industry_ppt_map = {}

    for ind_label in tqdm(industry_list, desc="分行业回归PPT"):
        ind_data = data[data['ind_label'] == ind_label].copy()
        months = sorted(ind_data['year_month'].unique())

        if len(months) < 12:
            continue

        # 逐月截面回归
        monthly_reg_results = []
        for month in months:
            month_data = ind_data[ind_data['year_month'] == month]
            if len(month_data) < 10:
                continue

            X = sm.add_constant(month_data[vars_model])
            y = month_data['exret_next']
            try:
                model = sm.OLS(y, X).fit()
                # 只需要log_prc的系数来算PPT
                month_result = {'log_prc': model.params.get('log_prc', np.nan)}
                monthly_reg_results.append(month_result)
            except Exception:
                continue

        if len(monthly_reg_results) < 6:
            continue

        # 计算该行业的PPT
        reg_results_df = pd.DataFrame(monthly_reg_results)
        mean_log_prc_coef = reg_results_df['log_prc'].mean()
        ind_ppt = -mean_log_prc_coef

        if np.isnan(ind_ppt) or ind_ppt <= 0:
            continue

        industry_ppt_map[ind_label] = ind_ppt
        print(f"  行业【{ind_label}】完成：PPT(φ) = {ind_ppt:.6f}，有效回归月份：{len(monthly_reg_results)}")

    print("\n" + "-" * 60)
    print(f"分行业PPT计算完成，有效行业数量：{len(industry_ppt_map)} 个")
    print("-" * 60)

    return industry_ppt_map


# ==========================================
# 核心3：估计月度滚动的 Beta 与 Alpha
# ==========================================
def estimate_classic_betas(monthly_df, factor_df, factor_cols, model_name):
    """
    对齐原作者逻辑：使用前60个月的月度数据，滚动预测下个月的 Beta、Alpha 和残差方差
    """
    print("\n" + "=" * 60)
    print(f"【{model_name}】步骤3：经典因子模型滚动估计(60个月) Beta 与 Alpha")
    data = pd.merge(monthly_df[['permno', 'year_month', 'exret']],
                    factor_df[['year_month'] + factor_cols],
                    on='year_month', how='inner')
    data = data.sort_values(['permno', 'year_month']).reset_index(drop=True)
    permnos = data['permno'].unique()
    beta_results = []
    for permno in tqdm(permnos, desc=f"{model_name} 月度滚动OLS估计"):
        stock_data = data[data['permno'] == permno].reset_index(drop=True)
        yms = stock_data['year_month'].values
        for i in range(len(yms) - ROLLING_BETA_WINDOW):
            sample = stock_data.iloc[i: i + ROLLING_BETA_WINDOW]
            next_ym = yms[i + ROLLING_BETA_WINDOW]
            if len(sample.dropna()) < 36:
                continue
            X = sm.add_constant(sample[factor_cols])
            y = sample['exret']
            try:
                model = sm.OLS(y, X).fit()
                res = {
                    'permno': permno,
                    'year_month': next_ym,
                    'alpha': model.params['const'],
                    'residual_var': np.var(model.resid, ddof=len(factor_cols) + 1)
                }
                # 修正语法错误
                for f in factor_cols:
                    res[f'beta_{f}'] = model.params.get(f, np.nan)
                beta_results.append(res)
            except Exception:
                continue
    beta_df = pd.DataFrame(beta_results).dropna()
    print(f"【{model_name}】月度Beta与Alpha估计完成，共生成 {len(beta_df)} 条记录")
    return beta_df


# ==========================================
# 核心4：【修改点3】均值-方差优化与分行业需求弹性计算
# ==========================================
def solve_qp_osqp(cov_matrix, mean_vector, tolerance=OPTIMIZATION_TOL):
    """OSQP求解带卖空约束的均值-方差优化"""
    n = len(mean_vector)
    P = sparse.csc_matrix(GAMMA * cov_matrix)
    q = -mean_vector.astype(np.float64)
    A = sparse.eye(n, format='csc')
    l = np.zeros(n)
    u = np.full(n, np.inf)
    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, eps_abs=1e-14, eps_rel=1e-14, max_iter=100000, verbose=False)
    res = solver.solve()
    if res.info.status != 'solved':
        return None, None
    weights = res.x
    positive_mask = weights > tolerance
    weights[~positive_mask] = 0.0
    return weights, positive_mask


def compute_classic_model_elasticity(beta_df, factor_df, industry_ppt_map, factor_cols, model_name, monthly_df):
    """
    适配分行业PPT的弹性计算：参考基准模型逻辑，计算个股弹性后再分组统计
    """
    print("\n" + "=" * 60)
    print(f"【{model_name}】步骤4：构建结构协方差、预期收益及计算需求弹性（适配分行业PPT）")

    mu_f = factor_df[factor_cols].mean().values
    Omega = factor_df[factor_cols].cov().values
    beta_cols = [f'beta_{f}' for f in factor_cols]

    results = []
    detailed_holdings = []

    # 行业-月度映射表
    stock_ind_map = monthly_df[['year_month', 'permno', 'ind_label']].drop_duplicates()

    months = sorted(beta_df['year_month'].unique())
    for ym in tqdm(months, desc=f"{model_name} 逐月优化与计算弹性"):
        month_data = beta_df[beta_df['year_month'] == ym].copy()
        if len(month_data) < 10:
            continue
        valid_stocks = month_data['permno'].values
        n_stocks = len(valid_stocks)

        # 匹配行业标签信息 (确保顺序一致)
        month_ind_info = stock_ind_map[
            (stock_ind_map['year_month'] == ym) &
            (stock_ind_map['permno'].isin(valid_stocks))
            ].set_index('permno').reindex(valid_stocks)

        month_data = month_data.set_index('permno').join(month_ind_info['ind_label']).reset_index()

        # 提取因子数据
        alpha = month_data['alpha'].values
        B = month_data[beta_cols].values.astype(np.float64)
        D = np.diag(month_data['residual_var'].values.astype(np.float64))

        Sigma = B @ Omega @ B.T + D
        mu = B @ mu_f  # 因子模型预期收益

        # 优化求解
        weights, positive_mask = solve_qp_osqp(Sigma, mu)
        if weights is None or np.sum(positive_mask) == 0:
            continue

        # 计算 Tau 和 UR
        cov_hold = Sigma[np.ix_(positive_mask, positive_mask)]
        try:
            inv_cov_hold = np.linalg.inv(cov_hold)
        except np.linalg.LinAlgError:
            inv_cov_hold = np.linalg.pinv(cov_hold)

        tau_hold = np.diag(inv_cov_hold)
        weights_hold = weights[positive_mask]
        ur_hold = weights_hold / tau_hold

        # 【核心逻辑】匹配持仓个股的行业与对应PPT
        held_permnos = valid_stocks[positive_mask]
        held_ind_label = month_data.loc[positive_mask, 'ind_label'].values

        # 逐个匹配行业PPT
        held_ppt = np.array([industry_ppt_map.get(ind, np.nan) for ind in held_ind_label])
        valid_ppt_mask = ~np.isnan(held_ppt) & (held_ppt > 0) & (ur_hold > 0)

        # 过滤无效样本
        held_permnos = held_permnos[valid_ppt_mask]
        held_ind_label = held_ind_label[valid_ppt_mask]
        weights_hold = weights_hold[valid_ppt_mask]
        tau_hold = tau_hold[valid_ppt_mask]
        ur_hold = ur_hold[valid_ppt_mask]
        held_ppt = held_ppt[valid_ppt_mask]

        # 计算个股弹性
        stock_elasticity = 1 + (held_ppt / ur_hold)

        if len(held_permnos) < 10: continue

        # 保存个股明细
        held_df = pd.DataFrame({
            'year_month': ym,
            'permno': held_permnos,
            'ind_label': held_ind_label,
            'weight': weights_hold,
            'tau': tau_hold,
            'ur': ur_hold,
            'industry_ppt': held_ppt,
            'stock_demand_elasticity': stock_elasticity
        })
        detailed_holdings.append(held_df)

        # ================= 统计汇总 =================
        # 全市场（用个股弹性平均）
        total_avg_ur = np.mean(ur_hold)
        total_avg_elasticity = np.mean(stock_elasticity)

        month_result = {
            'year_month': ym,
            'n_stocks_available': n_stocks,
            'n_hold_total': len(held_permnos),
            'avg_monthly_ur_total': total_avg_ur,
            'demand_elasticity_total': total_avg_elasticity
        }

        # 按行业分组统计
        grouped = held_df.groupby('ind_label', dropna=True)
        for ind_label, group in grouped:
            n = len(group)
            if n >= 2:
                ur = group['ur'].mean()
                elas = group['stock_demand_elasticity'].mean()
            else:
                ur = np.nan
                elas = np.nan

            month_result[f'n_ind_{ind_label}'] = n
            month_result[f'ur_ind_{ind_label}'] = ur
            month_result[f'elas_ind_{ind_label}'] = elas

        results.append(month_result)

    results_df = pd.DataFrame(results)
    results_df = results_df[results_df['n_hold_total'] >= 10].reset_index(drop=True)

    # ===================== 新增：剔除极端弹性月份（总弹性/行业弹性>2000直接删除） =====================
    # 获取所有分行业弹性列 + 总弹性列
    elasticity_cols = ['demand_elasticity_total'] + [c for c in results_df.columns if c.startswith('elas_ind_')]
    # 构建掩码：任意弹性超过阈值的月份
    extreme_mask = results_df[elasticity_cols].gt(MAX_ALLOWED_ELASTICITY).any(axis=1)
    # 剔除极端月份
    results_df = results_df[~extreme_mask].reset_index(drop=True)
    # ==============================================================================================

    # ==========================================
    # 最终汇总打印
    # ==========================================
    # ==========================================
    print("\n" + "=" * 60)
    print(f"===== {model_name} 模型最终结果 (含行业异质性) =====")
    print(f"有效测试月份: {len(results_df)}")
    print(f"【全市场】 平均持仓数: {results_df['n_hold_total'].mean():.1f} | "
          f"平均UR: {results_df['avg_monthly_ur_total'].mean():.6f} | "
          f"平均弹性: {results_df['demand_elasticity_total'].mean():.2f}")

    elas_cols = [c for c in results_df.columns if c.startswith('elas_ind_')]
    print("\n----- 分行业结果 -----")
    summary_data = []

    for col in sorted(elas_cols):
        ind_code = col.replace('elas_ind_', '')
        ur_col = f'ur_ind_{ind_code}'
        n_col = f'n_ind_{ind_code}'

        avg_n = results_df[n_col].mean()
        avg_ur = results_df[ur_col].mean()
        avg_elas = results_df[col].mean()
        valid_months = results_df[col].count()

        summary_data.append({
            'Industry': ind_code,
            'Valid_Months': valid_months,
            'Avg_Holdings': avg_n,
            'Avg_UR': avg_ur,
            'Avg_Elasticity': avg_elas
        })

        print(f"【行业{ind_code}】 有效月数:{valid_months} | 平均持仓数: {avg_n:.1f} | "
              f"平均UR: {avg_ur:.6f} | 平均弹性: {avg_elas:.2f}")

    print("=" * 60)

    industry_summary = pd.DataFrame(summary_data)
    detailed_df = pd.concat(detailed_holdings, ignore_index=True) if detailed_holdings else pd.DataFrame()
    avg_ur = results_df['avg_monthly_ur_total'].mean()
    avg_elasticity = results_df['demand_elasticity_total'].mean()

    return results_df, avg_ur, avg_elasticity, industry_summary, detailed_df


# ==========================================
# 主运行流
# ==========================================
def main_classic_factor_models():
    print("=" * 80)
    print("《Why do portfolio choice models predict inelastic demand?》")
    print("经典因子模型复现管线启动（含行业异质性+分行业PPT）")
    print("=" * 80)
    config = {
        'monthly_data_path': 'monthly_stock_returns_with_rf1.csv',
        'monthly_data_dir': '.',
        'factor_data_path': 'fivefactor.csv',
        'output_root_dir': 'classic_factor_models_results_industry'
    }
    os.makedirs(config['output_root_dir'], exist_ok=True)
    summary_results = []

    try:
        # 1. 全量数据加载
        monthly_df, factor_df, _ = load_and_clean_data(config['monthly_data_path'], config['factor_data_path'],
                                                       config['monthly_data_dir'])
        # 2. 分行业 PPT 系数计算
        industry_ppt_map = run_fama_macbeth_by_industry(monthly_df)
        if len(industry_ppt_map) == 0:
            raise ValueError("无有效行业的PPT计算结果，请检查数据")

        # 循环跑所有配置的因子模型
        for model_name, factor_cols in MODELS_CONFIG.items():
            print(f"\n\n>>>>>>>>>> 开始运行【{model_name}】模型 <<<<<<<<<<")
            model_output_dir = os.path.join(config['output_root_dir'], model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            # 3. 滚动回归Beta和Alpha
            beta_df = estimate_classic_betas(monthly_df, factor_df, factor_cols, model_name)

            # 4. 优化求解、计算弹性
            results_df, avg_ur, avg_elasticity, industry_summary, detailed_df = compute_classic_model_elasticity(
                beta_df, factor_df, industry_ppt_map, factor_cols, model_name, monthly_df
            )

            # 5. 结果落盘
            results_df.to_csv(os.path.join(model_output_dir, f'{model_name}_monthly_results.csv'), index=False)
            beta_df.to_csv(os.path.join(model_output_dir, f'{model_name}_betas.csv'), index=False)
            industry_summary.to_csv(os.path.join(model_output_dir, f'{model_name}_industry_summary.csv'), index=False)
            detailed_df.to_csv(os.path.join(model_output_dir, f'{model_name}_detailed_holdings.csv'), index=False)

            print(f"【{model_name}】所有复现结果已保存至：{model_output_dir}")

            summary_results.append({
                'model_name': model_name,
                'factor_cols': ','.join(factor_cols),
                'valid_months': len(results_df),
                'avg_monthly_unspanned_return': avg_ur,
                'avg_demand_elasticity': avg_elasticity
            })

        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(os.path.join(config['output_root_dir'], 'all_models_summary.csv'), index=False)
        print("\n\n" + "=" * 80)
        print("所有模型运行完成！全模型汇总结果：")
        print(summary_df.round(6))
        print("=" * 80)

    except Exception as e:
        print(f"程序执行失败：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_classic_factor_models()