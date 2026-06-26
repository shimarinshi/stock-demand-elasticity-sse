import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings
import os
import glob
from tqdm import tqdm
import osqp
from scipy import sparse
import gc

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==========================================
# 全局配置与参数（基准模型 Benchmark Model）
# ==========================================
MARKETTYPE_SSE = 1  # 上证A股代码
MIN_OBS_FRAC = 0.8  # 计算协方差时，股票日度数据至少需有80%的观测值
OPTIMIZATION_TOL = 1e-10


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
# 核心1：加载月度与日度数据
# ==========================================
def load_benchmark_data(monthly_data_path, daily_data_dir, monthly_data_dir='.'):
    """
    加载并清洗月度数据（财务特征）和日度收益率（仅用于协方差）
    """
    print("=" * 60)
    print("步骤1：加载上证A股月度特征数据与日度收益率")
    print("=" * 60)

    # 1. 加载月度数据
    monthly_df = load_data(monthly_data_path, "月度收益率数据")
    if 'yearmonth' in monthly_df.columns and 'year_month' not in monthly_df.columns:
        monthly_df = monthly_df.rename(columns={'yearmonth': 'year_month'})

    leverage_df = load_data(os.path.join(monthly_data_dir, 'monthly_leverage.csv'), "杠杆率数据")
    roa_investment_df = load_data(os.path.join(monthly_data_dir, 'monthly_roa_investment.csv'), "ROA数据")
    asset_growth_df = load_data(os.path.join(monthly_data_dir, 'monthly_asset_growth.csv'), "资产增长数据")

    # 合并财务特征
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

    # 过滤上证A股，清洗市值后20%
    monthly_df = monthly_df[monthly_df['Markettype'] == MARKETTYPE_SSE].reset_index(drop=True)
    monthly_df['permno'] = format_permno(monthly_df['permno'])
    monthly_df = monthly_df.dropna(subset=['market_equity', 'exret'])

    def filter_and_label_market_cap(group):
        # 【修改点1】不再剔除后20%，直接对全量股票处理
        valid_group = group.copy()

        # (可选保留) 市值分组标签可以保留，不影响行业分析
        valid_group['size_group'] = pd.qcut(
            valid_group['market_equity'],
            q=3,
            labels=['Small', 'Medium', 'Large'],
            duplicates='drop'
        )
        return valid_group

    monthly_df = monthly_df.groupby('year_month').apply(filter_and_label_market_cap).reset_index(drop=True)
    sse_permnos = monthly_df['permno'].unique()

    # [在 monthly_df 处理完筛选逻辑后添加]
    industry_lookup = load_data('industry_num.csv', name="行业分类数据")
    if not industry_lookup.empty:
        industry_lookup['permno'] = format_permno(industry_lookup['Stkcd'])
        industry_lookup['ind_label'] = industry_lookup['Nnindcd'].str[0].str.upper()
        # 合并到主表
        monthly_df = pd.merge(monthly_df, industry_lookup[['permno', 'ind_label']], on='permno', how='left')

    print(f"上证A股月度样本筛选完成，共 {len(sse_permnos)} 只股票，{len(monthly_df)} 条月度记录")

    # 2. 加载日度数据 (无需无风险利率，只需计算协方差)
    print("\n加载上证A股日度收益率数据...")
    daily_files = glob.glob(os.path.join(daily_data_dir, '*dailyret*.csv'))
    if not daily_files:
        raise ValueError("未找到日度收益率文件")

    daily_ret_list = []
    for f in tqdm(daily_files, desc="读取日度收益率文件"):
        df = load_data(f, name=f"日度文件{f}")
        if df.empty: continue
        col_map = {'Stkcd': 'permno', 'Trddt': 'date', 'Dretwd': 'ret'}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if not all(col in df.columns for col in ['permno', 'date', 'ret']): continue

        df['permno'] = format_permno(df['permno'])
        df = df[df['permno'].isin(sse_permnos)]
        if df.empty: continue

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
        df = df.dropna(subset=['permno', 'date', 'ret'])
        daily_ret_list.append(df[['permno', 'date', 'ret']])

    daily_ret_df = pd.concat(daily_ret_list, ignore_index=True).drop_duplicates(['permno', 'date'])
    daily_ret_df = daily_ret_df.sort_values(['permno', 'date']).reset_index(drop=True)

    # 构建年月的索引映射以便滚动截取过去12个月
    daily_ret_df['year_month'] = daily_ret_df['date'].dt.year * 100 + daily_ret_df['date'].dt.month

    print(f"日度数据加载完成，共 {len(daily_ret_df)} 条记录")
    return monthly_df, daily_ret_df


# ==========================================
# 核心2：特征回归：计算期望收益率 μ_t 与 PPT φ
# ==========================================
# ==========================================
# 核心2：【修改点2】分行业Fama-MacBeth回归计算PPT与预期收益率
# ==========================================
def run_fama_macbeth_by_industry(df):
    """
    按行业(ind_label)分组，每个行业单独跑Fama-MacBeth回归
    输出：1. 每个行业的专属PPT；2. 个股匹配所属行业的预期收益率
    """
    print("\n" + "=" * 60)
    print("步骤2：分行业Fama-MacBeth回归 计算行业专属PPT(φ) 及 预期收益率(μ_t)")
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
    industry_ppt_map = {}  # 存储：行业代码 -> PPT
    industry_coefs_map = {}  # 存储：行业代码 -> 特征溢价系数
    all_expected_return = []  # 存储所有个股的预期收益率

    for ind_label in tqdm(industry_list, desc="分行业Fama-MacBeth迭代"):
        ind_data = data[data['ind_label'] == ind_label].copy()
        months = sorted(ind_data['year_month'].unique())

        if len(months) < 12:  # 该行业有效月份不足1年，跳过
            continue

        # 逐月截面回归
        monthly_reg_results = []
        for month in months:
            month_data = ind_data[ind_data['year_month'] == month]
            if len(month_data) < 10:  # 【修改点3】当月该行业至少10只股票才做回归
                continue

            X = sm.add_constant(month_data[vars_model])
            y = month_data['exret_next']
            try:
                model = sm.OLS(y, X).fit()
                month_result = {'const': model.params.get('const', np.nan)}
                for v in vars_model:
                    month_result[v] = model.params.get(v, np.nan)
                monthly_reg_results.append(month_result)
            except Exception:
                continue

        if len(monthly_reg_results) < 6:  # 有效回归月份不足6个，跳过
            continue

        # 计算该行业的特征溢价均值与PPT
        reg_results_df = pd.DataFrame(monthly_reg_results)
        mean_coefs = reg_results_df.mean()
        ind_ppt = -mean_coefs.get('log_prc', np.nan)

        if np.isnan(ind_ppt) or ind_ppt <= 0:
            continue

        # 保存结果
        industry_ppt_map[ind_label] = ind_ppt
        industry_coefs_map[ind_label] = mean_coefs
        print(f"行业【{ind_label}】完成：PPT(φ) = {ind_ppt:.6f}，有效回归月份：{len(monthly_reg_results)}")

        # 用该行业专属系数计算该行业个股的预期收益率
        ind_data['expected_return'] = mean_coefs['const']
        for v in vars_model:
            ind_data['expected_return'] += ind_data[v] * mean_coefs[v]

        all_expected_return.append(ind_data[['year_month', 'permno', 'ind_label', 'expected_return']])

    # 合并结果
    expected_return_df = pd.concat(all_expected_return, ignore_index=True).dropna().reset_index(drop=True)

    print("\n" + "-" * 60)
    print(f"分行业回归完成，有效行业数量：{len(industry_ppt_map)} 个")
    print("-" * 60)

    return industry_ppt_map, expected_return_df


# ==========================================
# 核心3：计算滚动日度协方差与 Ledoit-Wolf 压缩
# ==========================================
def compute_rolling_covariance(monthly_df, daily_ret_df):
    """
    对齐 R 代码 `p.getOne` 中的 Ledoit-Wolf 收缩及缺值插补。
    """
    print("\n" + "=" * 60)
    print("步骤3：计算基于过去12个月日度收益率的 Ledoit-Wolf 协方差矩阵")

    unique_months = sorted(monthly_df['year_month'].unique())
    unique_daily_months = sorted(daily_ret_df['year_month'].unique())

    cov_results = []

    for i, target_ym in enumerate(tqdm(unique_months, desc="逐月计算日度协方差及收缩")):
        # 寻找对应的过去12个月
        try:
            target_idx = unique_daily_months.index(target_ym)
            if target_idx < 12: continue
            past_12_months = unique_daily_months[target_idx - 12: target_idx]
        except ValueError:
            continue

        valid_permnos = monthly_df[monthly_df['year_month'] == target_ym]['permno'].unique()

        # 获取过去12个月的日度数据，只保留当前月有效的股票
        window_daily = daily_ret_df[
            (daily_ret_df['year_month'].isin(past_12_months)) &
            (daily_ret_df['permno'].isin(valid_permnos))
            ]

        if window_daily.empty: continue

        # 转为宽表 (date为索引, permno为列)
        pivot_daily = window_daily.pivot(index='date', columns='permno', values='ret')

        # 筛选至少有 80% 观测值的股票
        max_obs = len(pivot_daily)
        valid_cols = pivot_daily.columns[pivot_daily.count() >= (max_obs * MIN_OBS_FRAC)]
        pivot_daily = pivot_daily[valid_cols]

        if pivot_daily.shape[1] < 5: continue

        # 缺失值填补：使用截面均值 (与R代码 `out[is.na(ret), ret := m]` 对齐)
        cross_sectional_mean = pivot_daily.mean(axis=1)
        pivot_daily = pivot_daily.apply(lambda col: col.fillna(cross_sectional_mean), axis=0)

        # -- Ledoit-Wolf (LW) 最优收缩逻辑 --
        Y = pivot_daily.values
        n_obs, p_vars = Y.shape
        if p_vars <= 1 or n_obs <= 1: continue

        # 样本协方差 (R代码: (t(Y) %*% Y)/n, 使用n而非n-1计算 LW 系数更为稳定)
        # 这里为保持精确，将均值归零(一般日度收益率均值极小)
        sample_cov = np.cov(Y, rowvar=False, bias=True)

        # 目标收缩矩阵: 常数协方差矩阵 (Constant Covariance)
        meanvar = np.mean(np.diag(sample_cov))
        meancov = (np.sum(sample_cov) - np.sum(np.diag(sample_cov))) / (p_vars * (p_vars - 1))
        target = (meanvar - meancov) * np.eye(p_vars) + meancov * np.ones((p_vars, p_vars))

        # LW公式计算收缩强度 kappa (按 Ledoit & Wolf 2004)
        Y_demeaned = Y - np.mean(Y, axis=0)
        sample2 = (Y_demeaned ** 2).T @ (Y_demeaned ** 2) / n_obs
        pi_mat = sample2 - sample_cov ** 2
        pi_hat = np.sum(pi_mat)

        gamma_hat = np.sum((sample_cov - target) ** 2)

        rho_diag = np.sum(np.diag(sample2)) / p_vars - np.sum(np.diag(sample_cov) ** 2) / p_vars
        sum1 = np.sum(Y_demeaned, axis=1)
        sum2 = np.sum(Y_demeaned ** 2, axis=1)
        rho_off1 = np.sum((sum1 ** 2 - sum2) ** 2) / p_vars / n_obs
        rho_off2 = (np.sum(sample_cov) - np.sum(np.diag(sample_cov))) ** 2 / p_vars
        rho_off = (rho_off1 - rho_off2) / (p_vars - 1)

        rho_hat = rho_diag + rho_off
        kappa_hat = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0
        shrinkage = max(0, min(1, kappa_hat / n_obs))

        # 最终协方差矩阵 (应用收缩，并乘以21月度化)
        final_cov = (1 - shrinkage) * sample_cov + shrinkage * target
        annualized_cov = final_cov * 21

        cov_results.append({
            'year_month': target_ym,
            'valid_stocks': pivot_daily.columns.tolist(),
            'cov_matrix': annualized_cov,
            'shrinkage_val': shrinkage
        })

    print(f"协方差矩阵计算完成，共生成 {len(cov_results)} 条月度记录")
    return cov_results


# ==========================================
# 核心4：均值-方差优化与基准模型需求弹性
# ==========================================
def solve_qp_osqp(cov_matrix, mean_vector, tolerance=OPTIMIZATION_TOL):
    """
    OSQP 二次规划求解（对齐 R 代码 solve_qp_osqp）。
    受限于 w >= 0 的非负卖空约束。
    """
    n = len(mean_vector)
    # OSQP 求解 min(0.5 x^T P x + q^T x)
    P = sparse.csc_matrix(cov_matrix)
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


def compute_benchmark_ur_and_elasticity(cov_results, expected_return_df, industry_ppt_map, monthly_df):
    """
    【修改点4】适配分行业PPT的弹性计算
    """
    print("\n" + "=" * 60)
    print("步骤4：基准模型均值-方差优化、UR计算与需求弹性提取（适配分行业PPT）")

    results = []
    detailed_holdings = []
    # 行业-月度映射表
    stock_ind_map = monthly_df[['year_month', 'permno', 'ind_label']].drop_duplicates()

    for cov_item in tqdm(cov_results, desc="逐月优化与计算弹性"):
        ym = cov_item['year_month']
        valid_stocks = cov_item['valid_stocks']
        cov_matrix = cov_item['cov_matrix']

        # 匹配该月的预期收益率和行业标签
        month_mu = expected_return_df[
            (expected_return_df['year_month'] == ym) &
            (expected_return_df['permno'].isin(valid_stocks))
            ].set_index('permno').reindex(valid_stocks)

        # 过滤无效样本
        valid_mask = ~month_mu['expected_return'].isna()
        if not valid_mask.any(): continue

        clean_mu = month_mu.loc[valid_mask, 'expected_return'].values
        clean_cov = cov_matrix[np.ix_(valid_mask.values, valid_mask.values)]
        clean_permnos = np.array(valid_stocks)[valid_mask.values]
        clean_ind_label = month_mu.loc[valid_mask, 'ind_label'].values

        # 优化求解
        weights, pos_mask = solve_qp_osqp(clean_cov, clean_mu)
        if weights is None or np.sum(pos_mask) == 0: continue

        # 计算 Tau 和 UR
        cov_hold = clean_cov[np.ix_(pos_mask, pos_mask)]
        try:
            inv_cov_hold = np.linalg.inv(cov_hold)
        except np.linalg.LinAlgError:
            inv_cov_hold = np.linalg.pinv(cov_hold)

        tau_hold = np.diag(inv_cov_hold)
        weights_hold = weights[pos_mask]
        ur_hold = weights_hold / tau_hold

        # 【修改点5】匹配持仓个股的行业与对应PPT，计算个股弹性
        held_permnos = clean_permnos[pos_mask]
        held_ind_label = clean_ind_label[pos_mask]

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
        stock_elasticity = 1 + (held_ppt / ur_hold)  # 个股弹性

        if len(held_permnos) < 10: continue

        # 构建个股明细
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

        # 构建本月结果字典
        month_result = {
            'year_month': ym,
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
    # ===================== 新增：排除200507月份(数据有极端值)，不参与任何平均 =====================
    results_df = results_df[results_df['year_month'] != 200507].reset_index(drop=True)
    detailed_df = pd.concat(detailed_holdings, ignore_index=True)
    detailed_df = detailed_df[detailed_df['year_month'] != 200507].reset_index(drop=True)

    # ==========================================
    # 最终汇总打印
    # ==========================================
    print("\n" + "=" * 60)
    print("===== 分行业需求弹性汇总 (Cross-month Average) =====")

    elas_cols = [c for c in results_df.columns if c.startswith('elas_ind_')]
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
            'Elasticity': avg_elas
        })

    industry_summary = pd.DataFrame(summary_data)
    print(industry_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}" if x >= 1 else f"{x:.6f}"))
    print("=" * 60)

    return results_df, industry_summary, detailed_df


# ==========================================
# 主运行流
# ==========================================
def main_benchmark_model():
    print("=" * 80)
    print("《Why do portfolio choice models predict inelastic demand?》")
    print("Benchmark 基准模型 (特征预期收益 + 压缩日度协方差)复现管线启动")
    print("=" * 80)

    config = {
        'monthly_data_path': 'monthly_stock_returns_with_rf1.csv',
        'monthly_data_dir': '.',
        'daily_data_dir': '.',
        'output_dir': 'benchmark_model_results'
    }
    os.makedirs(config['output_dir'], exist_ok=True)

    try:
        # 1. 加载月度和日度数据
        monthly_df, daily_ret_df = load_benchmark_data(
            config['monthly_data_path'],
            config['daily_data_dir'],
            config['monthly_data_dir']
        )

        industry_ppt_map, expected_return_df = run_fama_macbeth_by_industry(monthly_df)
        if len(industry_ppt_map) == 0:
            raise ValueError("无有效行业的PPT计算结果，请检查行业数据")

        # 3. 计算基于过去12个月日度收益的 Ledoit-Wolf 缩水协方差矩阵 (年化)
        cov_results = compute_rolling_covariance(monthly_df, daily_ret_df)

        # 4. OSQP 均值方差优化提取 UR 进而推算弹性
        results_df, industry_summary, detailed_df = compute_benchmark_ur_and_elasticity(
            cov_results, expected_return_df, industry_ppt_map, monthly_df
        )

        # 保存结果
        results_df.to_csv(os.path.join(config['output_dir'], 'monthly_industry_ur.csv'), index=False)
        industry_summary.to_csv(os.path.join(config['output_dir'], 'industry_heterogeneity_summary.csv'), index=False)
        detailed_df.to_csv(os.path.join(config['output_dir'], 'detailed_holdings.csv'), index=False)
        print(f"所有结果已保存至：{config['output_dir']}")

    except Exception as e:
        print(f"程序执行失败：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_benchmark_model()
