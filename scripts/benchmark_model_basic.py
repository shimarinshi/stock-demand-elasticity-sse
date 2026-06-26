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

    def filter_by_market_cap(group):
        q20 = group['market_equity'].quantile(0.2)
        return group[group['market_equity'] > q20]

    monthly_df = monthly_df.groupby('year_month').apply(filter_by_market_cap).reset_index(drop=True)
    sse_permnos = monthly_df['permno'].unique()
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
# 核心2：特征回归：计算期望收益率 $\mu_t$ 与 PPT $\phi$
# ==========================================
def run_fama_macbeth_for_mu_and_ppt(df):
    """
    对齐 R 代码 `1_data_and_forecasting.R`。
    计算PPT（φ），同时生成全样本平均系数下的预期收益率 Exrethat。
    """
    print("\n" + "=" * 60)
    print("步骤2：Fama-MacBeth 回归计算 PPT(φ) 及 预期收益率(μ_t)")
    data = df.copy()
    data = data[(data['prc'] > 0) & (data['vol'] > 0) & (data['market_equity'] > 0)]

    data['log_prc'] = np.log(data['prc'].abs())
    data['log_prc_lag'] = data.groupby('permno')['log_prc'].shift(1)
    data['log_vol'] = np.log(data['vol'])
    data['log_me'] = np.log(data['market_equity'])
    data = data.replace([np.inf, -np.inf], np.nan)

    data['std_log_vol'] = standardize_monthly('log_vol', 'year_month', data)
    data['std_log_me'] = standardize_monthly('log_me', 'year_month', data)

    # 尽力对齐论文特征的变量（动态适配现有的列）
    base_vars = ['log_prc', 'log_prc_lag', 'std_log_vol', 'std_log_me', 'monthly_roa', 'monthly_invest_return']
    vars_model = [v for v in base_vars if v in data.columns]

    for col in vars_model:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 回归标的为下一期的超额收益率
    data['exret_next'] = data.groupby('permno')['exret'].shift(-1)
    data = data.dropna(subset=['exret_next'] + vars_model)

    months = sorted(data['year_month'].unique())
    results = []
    for month in tqdm(months, desc="Fama-MacBeth 迭代(特征回归)"):
        month_data = data[data['year_month'] == month]
        if len(month_data) < 10: continue
        X = sm.add_constant(month_data[vars_model])
        y = month_data['exret_next']
        try:
            model = sm.OLS(y, X).fit()
            month_result = {'const': model.params.get('const', np.nan)}
            for v in vars_model:
                month_result[v] = model.params.get(v, np.nan)
            results.append(month_result)
        except Exception:
            continue

    results_df = pd.DataFrame(results)
    mean_coefs = results_df.mean()

    ppt_phi = -mean_coefs['log_prc']
    print(f"价格传递系数PPT(φ) 计算完成：φ = {ppt_phi:.6f}")

    # 使用全样本平均系数计算每个月的预期收益率 mu_t (对应 exrethat_all_vars)
    data['expected_return'] = mean_coefs['const']
    for v in vars_model:
        data['expected_return'] += data[v] * mean_coefs[v]

    expected_return_df = data[['year_month', 'permno', 'expected_return']].dropna().reset_index(drop=True)
    return ppt_phi, expected_return_df


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


def compute_benchmark_ur_and_elasticity(cov_results, expected_return_df, ppt_phi):
    """
    使用基准模型计算 Unspanned Return 和需求弹性 $\eta = 1 + \phi / UR$。
    """
    print("\n" + "=" * 60)
    print("步骤4：基准模型均值-方差优化、UR计算与需求弹性提取")

    results = []

    for cov_item in tqdm(cov_results, desc="逐月优化与计算弹性"):
        ym = cov_item['year_month']
        valid_stocks = cov_item['valid_stocks']
        cov_matrix = cov_item['cov_matrix']

        # 匹配该月的预期收益率
        month_mu = expected_return_df[
            (expected_return_df['year_month'] == ym) &
            (expected_return_df['permno'].isin(valid_stocks))
            ]
        month_mu = month_mu.set_index('permno').reindex(valid_stocks)

        # 若出现无预测收益的情况，丢弃对应股票并切片矩阵
        valid_mask = ~month_mu['expected_return'].isna()
        if not valid_mask.any(): continue

        clean_mu = month_mu.loc[valid_mask, 'expected_return'].values
        clean_cov = cov_matrix[np.ix_(valid_mask.values, valid_mask.values)]

        weights, pos_mask = solve_qp_osqp(clean_cov, clean_mu)
        if weights is None or np.sum(pos_mask) == 0: continue

        # ==========================================
        # 核心步骤：计算 UR = w_hold / tau_hold
        # 此处精确对齐 R 代码：invC = solve(C[idx, idx]); tau = diag(invC)
        # ==========================================
        cov_hold = clean_cov[np.ix_(pos_mask, pos_mask)]
        try:
            inv_cov_hold = np.linalg.inv(cov_hold)
        except np.linalg.LinAlgError:
            inv_cov_hold = np.linalg.pinv(cov_hold)

        tau_hold = np.diag(inv_cov_hold)
        weights_hold = weights[pos_mask]

        ur_hold = weights_hold / tau_hold
        avg_monthly_ur = np.mean(ur_hold)

        if avg_monthly_ur <= 0: continue

        demand_elasticity = 1 + (ppt_phi / avg_monthly_ur)

        results.append({
            'year_month': ym,
            'n_stocks_available': len(clean_mu),
            'n_hold': np.sum(pos_mask),
            'avg_monthly_ur': avg_monthly_ur,
            'demand_elasticity': demand_elasticity,
            'optimal_shrinkage_param': cov_item['shrinkage_val']
        })

    results_df = pd.DataFrame(results)

    # ==========================================
    # 【新增修改】过滤掉持仓数 n_hold 低于 10 的月份
    # ==========================================
    results_df = results_df[results_df['n_hold'] >= 10].reset_index(drop=True)

    print("\n" + "=" * 60)
    print("===== Benchmark 基准模型最终结果 =====")
    print(f"有效测试月份: {len(results_df)}")
    print(f"全样本平均需求弹性: {results_df['demand_elasticity'].mean():.2f}")
    print(f"全样本平均 Unspanned Return: {results_df['avg_monthly_ur'].mean():.6f}")
    print(f"全样本平均最优收缩参数 (Shrinkage): {results_df['optimal_shrinkage_param'].mean():.4f}")
    print("=" * 60)

    return results_df


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

        # 2. 通过 Fama-MacBeth 得到期望收益 mu_t 与弹性分子 phi
        ppt_phi, expected_return_df = run_fama_macbeth_for_mu_and_ppt(monthly_df)
        if np.isnan(ppt_phi): raise ValueError("PPT(φ)计算失败")

        # 3. 计算基于过去12个月日度收益的 Ledoit-Wolf 缩水协方差矩阵 (年化)
        cov_results = compute_rolling_covariance(monthly_df, daily_ret_df)

        # 4. OSQP 均值方差优化提取 UR 进而推算弹性
        results_df = compute_benchmark_ur_and_elasticity(cov_results, expected_return_df, ppt_phi)

        # 保存结果
        results_df.to_csv(os.path.join(config['output_dir'], 'benchmark_elasticity_results.csv'), index=False)
        expected_return_df.to_csv(os.path.join(config['output_dir'], 'benchmark_expected_returns.csv'), index=False)
        print(f"所有基准模型复现结果已保存至：{config['output_dir']}")

    except Exception as e:
        print(f"程序执行失败：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_benchmark_model()