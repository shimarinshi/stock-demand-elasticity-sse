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
# 全局配置与参数（最小化修改，新增多模型配置）
# ==========================================
MARKETTYPE_SSE = 1  # 上证A股代码
FACTOR_MARKET_ID = 'P9701'  # 上证A股因子市场编码
FACTOR_PORTFOLIO = 1  # 目标投资组合类型
ROLLING_BETA_WINDOW = 60  # 经典因子模型使用5年(60个月)的月度滚动窗口
GAMMA = 1  # 风险厌恶系数
OPTIMIZATION_TOL = 1e-10

# --------------------------
# 新增：多模型因子配置（完全满足FF3、CAPM、FF5需求）
# --------------------------
MODELS_CONFIG = {
    "CAPM": ['MKT'],  # 单因子市场模型
    "FF3": ['MKT', 'SMB', 'HML'],  # Fama-French三因子模型
    "FF5": ['MKT', 'SMB', 'HML', 'RMW', 'CMA'],  # 原Fama-French五因子模型
    "FF6": ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']  # 新增：FF5+动量因子的六因子模型
}
# 全量因子列名映射（兼容所有模型，不修改原数据读取逻辑）
ALL_FACTOR_COL_MAPPING = {'RiskPremium2': 'MKT', 'SMB2': 'SMB', 'HML2': 'HML', 'RMW2': 'RMW', 'CMA2': 'CMA'}


# ==========================================
# 基础工具函数（完全保留原代码，无修改）
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
# 核心1：加载月度数据与因子数据（无核心逻辑修改，兼容全量因子）
# ==========================================
def load_and_clean_data(monthly_data_path, factor_data_path, monthly_data_dir='.'):
    """
    加载并清洗月度收益数据与月度五因子数据（兼容所有因子模型）
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
    # 按月筛选市值前80%
    monthly_df = monthly_df.dropna(subset=['market_equity', 'exret'])

    def filter_by_market_cap(group):
        q20 = group['market_equity'].quantile(0.2)
        return group[group['market_equity'] > q20]

    monthly_df = monthly_df.groupby('year_month').apply(filter_by_market_cap).reset_index(drop=True)
    sse_permnos = monthly_df['permno'].unique()
    print(f"上证A股月度样本筛选完成，共 {len(sse_permnos)} 只股票，{len(monthly_df)} 条记录")
    # 2. 加载月度因子数据（全量加载，兼容所有模型）
    factor_df = load_data(factor_data_path, "月度因子数据")
    if 'TradingMonth' in factor_df.columns:
        factor_df['year_month'] = pd.to_datetime(factor_df['TradingMonth'], errors='coerce')
        factor_df['year_month'] = (factor_df['year_month'].dt.year * 100 + factor_df['year_month'].dt.month).fillna(
            0).astype(int)
    # 筛选市场与组合，并重命名全量因子
    factor_df = factor_df[
        (factor_df['MarkettypeID'] == FACTOR_MARKET_ID) &
        (factor_df['Portfolios'] == FACTOR_PORTFOLIO)
        ].reset_index(drop=True)
    factor_df = factor_df.rename(columns={k: v for k, v in ALL_FACTOR_COL_MAPPING.items() if k in factor_df.columns})
    # 保留所有因子列，按时间排序
    factor_df = factor_df[['year_month'] + list(ALL_FACTOR_COL_MAPPING.values())].dropna().sort_values(
        'year_month').reset_index(drop=True)
    mom_df = load_data('momfactor.csv', "月度动量因子数据")
    if not mom_df.empty:
        # 1. 时间列处理，和原有因子格式统一
        if 'TradingMonth' in mom_df.columns:
            mom_df['year_month'] = pd.to_datetime(mom_df['TradingMonth'], errors='coerce')
            mom_df['year_month'] = (mom_df['year_month'].dt.year * 100 + mom_df['year_month'].dt.month).fillna(
                0).astype(int)

        # 2. 严格按你的要求筛选数据
        mom_df = mom_df[
            (mom_df['MarkettypeID'] == FACTOR_MARKET_ID) &  # 和原有因子市场统一
            (mom_df['Quantile'] == '10%') &  # Quantile列值为10%
            (mom_df['StockClass'] == 0) &  # StockClass为0
            (mom_df['FormationPeriod'] == 12)  # 形成期12个月
            ].reset_index(drop=True)

        # 3. 重命名动量因子列，和其他因子命名统一
        mom_df = mom_df.rename(columns={'MomRe2': 'MOM'})

        # 4. 仅保留需要的列，和原有因子表合并
        mom_df = mom_df[['year_month', 'MOM']].dropna()
        factor_df = pd.merge(factor_df, mom_df, on='year_month', how='inner')
        print(f"动量因子数据加载完成，有效时间范围：{mom_df['year_month'].min()} 至 {mom_df['year_month'].max()}")
    else:
        raise ValueError("动量因子文件momfactor.csv加载失败，请检查文件路径与格式")
    # ====================== 新增：动量因子加载与处理 结束 ======================

    print(f"全量因子数据（含动量）加载完成，时间范围：{factor_df['year_month'].min()} 至 {factor_df['year_month'].max()}")
    return monthly_df, factor_df, sse_permnos


# ==========================================
# 核心2：计算价格传递系数 PPT (φ)（完全保留原代码，无修改，与因子模型无关）
# ==========================================
def run_fama_macbeth_for_ppt(df):
    """计算价格传递系数PPT（φ），回归预期收益仅用于提取系数，不进入优化组合"""
    print("\n" + "=" * 60)
    print("步骤2：Fama-MacBeth回归计算PPT（φ）")
    data = df.copy()
    data = data[(data['prc'] > 0) & (data['vol'] > 0) & (data['market_equity'] > 0)]
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
    data['exret_next'] = data.groupby('permno')['exret'].shift(-1)
    data = data.dropna(subset=['exret_next'] + vars_model)
    months = sorted(data['year_month'].unique())
    results = []
    for month in tqdm(months, desc="Fama-MacBeth回归迭代"):
        month_data = data[data['year_month'] == month]
        if len(month_data) < 10: continue
        X = sm.add_constant(month_data[vars_model])
        y = month_data['exret_next']
        try:
            model = sm.OLS(y, X).fit()
            month_result = {v: model.params.get(v, np.nan) for v in vars_model}
            results.append(month_result)
        except Exception:
            continue
    log_prc_coef = pd.DataFrame(results)['log_prc'].mean()
    ppt = -log_prc_coef
    print(f"价格传递系数PPT(φ)计算完成：φ = {ppt:.6f}")
    return ppt


# ==========================================
# 核心3：估计月度滚动的 Beta 与 Alpha（仅新增factor_cols入参，无逻辑修改）
# ==========================================
def estimate_classic_betas(monthly_df, factor_df, factor_cols, model_name):
    """
    对齐原作者逻辑：使用前60个月的月度数据，滚动预测下个月的 Beta、Alpha 和残差方差
    新增factor_cols入参，适配不同因子模型
    """
    print("\n" + "=" * 60)
    print(f"【{model_name}】步骤3：经典因子模型滚动估计(60个月) Beta 与 Alpha")
    # 将超额收益与因子合并，仅保留当前模型需要的因子列
    data = pd.merge(monthly_df[['permno', 'year_month', 'exret']],
                    factor_df[['year_month'] + factor_cols],
                    on='year_month', how='inner')
    data = data.sort_values(['permno', 'year_month']).reset_index(drop=True)
    permnos = data['permno'].unique()
    beta_results = []
    for permno in tqdm(permnos, desc=f"{model_name} 月度滚动OLS估计"):
        stock_data = data[data['permno'] == permno].reset_index(drop=True)
        yms = stock_data['year_month'].values
        # 严格使用过去60个月预测下个月
        for i in range(len(yms) - ROLLING_BETA_WINDOW):
            sample = stock_data.iloc[i: i + ROLLING_BETA_WINDOW]
            next_ym = yms[i + ROLLING_BETA_WINDOW]
            # 确保样本内至少有36个非空观测值以保证回归稳定
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
                for f in factor_cols:
                    res[f'beta_{f}'] = model.params.get(f, np.nan)
                beta_results.append(res)
            except Exception:
                continue
    beta_df = pd.DataFrame(beta_results).dropna()
    print(f"【{model_name}】月度Beta与Alpha估计完成，共生成 {len(beta_df)} 条记录")
    return beta_df


# ==========================================
# 核心4：均值-方差优化与需求弹性计算（仅新增入参，无逻辑修改，新增UR输出）
# ==========================================
def solve_qp_osqp(cov_matrix, mean_vector, tolerance=OPTIMIZATION_TOL):
    """OSQP求解带卖空约束的均值-方差优化（完全保留原代码）"""
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


def compute_classic_model_elasticity(beta_df, factor_df, ppt_phi, factor_cols, model_name):
    """
    经典因子的预期收益率与协方差结构计算，适配不同因子模型
    新增全样本平均Unspanned Return输出
    """
    print("\n" + "=" * 60)
    print(f"【{model_name}】步骤4：构建结构协方差、预期收益及计算需求弹性")
    # 提取当前模型因子的全样本期望均值与协方差
    mu_f = factor_df[factor_cols].mean().values
    Omega = factor_df[factor_cols].cov().values
    beta_cols = [f'beta_{f}' for f in factor_cols]

    results = []
    months = sorted(beta_df['year_month'].unique())
    for ym in tqdm(months, desc=f"{model_name} 逐月优化与计算弹性"):
        month_data = beta_df[beta_df['year_month'] == ym].copy()
        if len(month_data) < 10:
            continue
        valid_stocks = month_data['permno'].values
        n_stocks = len(valid_stocks)
        # 提取当前月的Alpha, Betas 和残差方差
        alpha = month_data['alpha'].values
        B = month_data[beta_cols].values.astype(np.float64)
        D = np.diag(month_data['residual_var'].values.astype(np.float64))
        # 经典因子模型的 Σ 与 μ
        Sigma = B @ Omega @ B.T + D
        mu = B @ mu_f
        # 二次规划求解权重
        weights, positive_mask = solve_qp_osqp(Sigma, mu)
        if weights is None or np.sum(positive_mask) == 0:
            continue
        # 仅针对持有资产计算 UR = w_hold / tau_hold
        cov_hold = Sigma[np.ix_(positive_mask, positive_mask)]
        try:
            inv_cov_hold = np.linalg.inv(cov_hold)
        except np.linalg.LinAlgError:
            inv_cov_hold = np.linalg.pinv(cov_hold)
        tau_hold = np.diag(inv_cov_hold)
        weights_hold = weights[positive_mask]
        ur_hold = weights_hold / tau_hold
        avg_monthly_ur = np.mean(ur_hold)
        if avg_monthly_ur <= 0:
            continue
        # 弹性计算
        demand_elasticity = 1 + (ppt_phi / avg_monthly_ur)
        results.append({
            'year_month': ym,
            'n_stocks': n_stocks,
            'n_hold': np.sum(positive_mask),
            'avg_monthly_ur': avg_monthly_ur,
            'demand_elasticity': demand_elasticity
        })
    results_df = pd.DataFrame(results)
    results_df = results_df[results_df['n_hold'] >= 10].reset_index(drop=True)
    # --------------------------
    # 新增：计算全样本平均Unspanned Return并打印
    # --------------------------
    avg_ur = results_df['avg_monthly_ur'].mean()
    avg_elasticity = results_df['demand_elasticity'].mean()
    print("\n" + "=" * 60)
    print(f"===== {model_name} 模型最终结果 =====")
    print(f"有效测试月份: {len(results_df)}")
    print(f"全样本平均月度Unspanned Return: {avg_ur:.6%}")  # 满足需求1
    print(f"全样本平均需求弹性: {avg_elasticity:.2f}")
    print("=" * 60)
    return results_df, avg_ur, avg_elasticity


# ==========================================
# 主运行流（新增多模型循环，最小化修改）
# ==========================================
def main_classic_factor_models():
    print("=" * 80)
    print("《Why do portfolio choice models predict inelastic demand?》")
    print("经典因子模型复现管线启动（CAPM/FF3/FF5三模型联动）")
    print("=" * 80)
    config = {
        'monthly_data_path': 'monthly_stock_returns_with_rf1.csv',
        'monthly_data_dir': '.',
        'factor_data_path': 'fivefactor.csv',
        'output_root_dir': 'classic_factor_models_results'
    }
    # 创建根输出文件夹
    os.makedirs(config['output_root_dir'], exist_ok=True)
    # 汇总结果表
    summary_results = []

    try:
        # 1. 全量数据加载（仅执行一次，所有模型共用）
        monthly_df, factor_df, _ = load_and_clean_data(config['monthly_data_path'], config['factor_data_path'],
                                                       config['monthly_data_dir'])
        # 2. PPT系数计算（仅执行一次，所有模型共用，与因子模型无关）
        ppt_phi = run_fama_macbeth_for_ppt(monthly_df)
        if np.isnan(ppt_phi):
            raise ValueError("PPT(φ)计算失败，程序终止")

        # --------------------------
        # 新增：循环跑所有配置的因子模型
        # --------------------------
        for model_name, factor_cols in MODELS_CONFIG.items():
            print(f"\n\n>>>>>>>>>> 开始运行【{model_name}】模型 <<<<<<<<<<")
            # 创建模型专属输出文件夹
            model_output_dir = os.path.join(config['output_root_dir'], model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            # 3. 滚动回归Beta和Alpha
            beta_df = estimate_classic_betas(monthly_df, factor_df, factor_cols, model_name)
            # 4. 优化求解、计算弹性与Unspanned Return
            results_df, avg_ur, avg_elasticity = compute_classic_model_elasticity(beta_df, factor_df, ppt_phi,
                                                                                  factor_cols, model_name)

            # 5. 模型结果落盘
            results_df.to_csv(os.path.join(model_output_dir, f'{model_name}_elasticity_results.csv'), index=False)
            beta_df.to_csv(os.path.join(model_output_dir, f'{model_name}_betas.csv'), index=False)
            print(f"【{model_name}】所有复现结果已保存至：{model_output_dir}")

            # 汇总结果
            summary_results.append({
                'model_name': model_name,
                'factor_cols': ','.join(factor_cols),
                'valid_months': len(results_df),
                'avg_monthly_unspanned_return': avg_ur,
                'avg_demand_elasticity': avg_elasticity
            })

        # 所有模型跑完后，输出汇总表
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(os.path.join(config['output_root_dir'], 'all_models_summary.csv'), index=False)
        print("\n\n" + "=" * 80)
        print("所有模型运行完成！全模型汇总结果：")
        print(summary_df.round(6))
        print(f"汇总结果已保存至：{os.path.join(config['output_root_dir'], 'all_models_summary.csv')}")
        print("=" * 80)

    except Exception as e:
        print(f"程序执行失败：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_classic_factor_models()