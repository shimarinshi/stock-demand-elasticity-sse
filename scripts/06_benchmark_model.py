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
MARKETTYPE_SSE = 1
MIN_OBS_FRAC = 0.8
OPTIMIZATION_TOL = 1e-10

# 【修改点 1】：在此设置固定收缩参数。若设为 None，则自动计算 Ledoit-Wolf 最优收缩
# 例如：FIXED_SHRINKAGE = 0.01 或 FIXED_SHRINKAGE = 0.9
#FIXED_SHRINKAGE = 0.05


# ==========================================
# 基础工具函数
# ==========================================
def format_permno(series):
    return series.astype(str).str.split('.').str[0].str.zfill(6)


def standardize_monthly(series, group_col, df):
    scaler = StandardScaler()

    def _standardize(group):
        group_clean = group.replace([np.inf, -np.inf], np.nan).dropna()
        if len(group_clean) < 5 or group_clean.std() < 1e-8:
            return group - group.mean()
        scaler.fit(group_clean.values.reshape(-1, 1))
        return pd.Series(scaler.transform(group.values.reshape(-1, 1)).flatten(), index=group.index)

    return df.groupby(group_col)[series].transform(_standardize)


def load_data(file_path, name="数据文件"):
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
    return pd.DataFrame()


# ==========================================
# 核心1：加载数据 (保持不变)
# ==========================================
def load_benchmark_data(monthly_data_path, daily_data_dir, monthly_data_dir='.'):
    print("=" * 60)
    print("步骤1：加载上证A股月度特征数据与日度收益率")
    print("=" * 60)
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
    monthly_df = monthly_df[monthly_df['Markettype'] == MARKETTYPE_SSE].reset_index(drop=True)
    monthly_df['permno'] = format_permno(monthly_df['permno'])
    monthly_df = monthly_df.dropna(subset=['market_equity', 'exret'])

    def filter_by_market_cap(group):
        q20 = group['market_equity'].quantile(0.2)
        return group[group['market_equity'] > q20]

    monthly_df = monthly_df.groupby('year_month').apply(filter_by_market_cap).reset_index(drop=True)
    sse_permnos = monthly_df['permno'].unique()
    daily_files = glob.glob(os.path.join(daily_data_dir, '*dailyret*.csv'))
    daily_ret_list = []
    for f in tqdm(daily_files, desc="读取日度收益率文件"):
        df = load_data(f, name=f"日度文件{f}")
        if df.empty: continue
        col_map = {'Stkcd': 'permno', 'Trddt': 'date', 'Dretwd': 'ret'}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df['permno'] = format_permno(df['permno'])
        df = df[df['permno'].isin(sse_permnos)]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
        df = df.dropna(subset=['permno', 'date', 'ret'])
        daily_ret_list.append(df[['permno', 'date', 'ret']])
    daily_ret_df = pd.concat(daily_ret_list, ignore_index=True).drop_duplicates(['permno', 'date'])
    daily_ret_df['year_month'] = daily_ret_df['date'].dt.year * 100 + daily_ret_df['date'].dt.month
    return monthly_df, daily_ret_df


# ==========================================
# 核心2：特征回归 (保持不变)
# ==========================================
def run_fama_macbeth_for_mu_and_ppt(df):
    data = df.copy()
    data = data[(data['prc'] > 0) & (data['vol'] > 0) & (data['market_equity'] > 0)]
    data['log_prc'] = np.log(data['prc'].abs())
    data['log_prc_lag'] = data.groupby('permno')['log_prc'].shift(1)
    data['log_vol'] = np.log(data['vol'])
    data['log_me'] = np.log(data['market_equity'])
    data['std_log_vol'] = standardize_monthly('log_vol', 'year_month', data)
    data['std_log_me'] = standardize_monthly('log_me', 'year_month', data)
    base_vars = ['log_prc', 'log_prc_lag', 'std_log_vol', 'std_log_me', 'monthly_roa', 'monthly_invest_return']
    vars_model = [v for v in base_vars if v in data.columns]
    data['exret_next'] = data.groupby('permno')['exret'].shift(-1)
    data = data.dropna(subset=['exret_next'] + vars_model)
    results = []
    for month in tqdm(sorted(data['year_month'].unique()), desc="Fama-MacBeth 迭代"):
        month_data = data[data['year_month'] == month]
        X = sm.add_constant(month_data[vars_model])
        y = month_data['exret_next']
        try:
            model = sm.OLS(y, X).fit()
            results.append({**{'const': model.params['const']}, **{v: model.params[v] for v in vars_model}})
        except:
            continue
    mean_coefs = pd.DataFrame(results).mean()
    ppt_phi = -mean_coefs['log_prc']
    data['expected_return'] = mean_coefs['const']
    for v in vars_model: data['expected_return'] += data[v] * mean_coefs[v]
    return ppt_phi, data[['year_month', 'permno', 'expected_return']].dropna().reset_index(drop=True)


# ==========================================
# 核心3：计算协方差 (增加固定收缩逻辑)
# ==========================================
def compute_rolling_covariance(monthly_df, daily_ret_df, fixed_shrinkage=None):
    """
    【修改点 2】：增加 fixed_shrinkage 参数判断
    """
    print("\n" + "=" * 60)
    msg = f"步骤3：计算协方差 (使用固定参数: {fixed_shrinkage})" if fixed_shrinkage is not None else "步骤3：计算协方差 (使用 Ledoit-Wolf 最优收缩)"
    print(msg)

    unique_months = sorted(monthly_df['year_month'].unique())
    unique_daily_months = sorted(daily_ret_df['year_month'].unique())
    cov_results = []

    for i, target_ym in enumerate(tqdm(unique_months, desc="逐月计算协方差")):
        try:
            target_idx = unique_daily_months.index(target_ym)
            if target_idx < 12: continue
            past_12_months = unique_daily_months[target_idx - 12: target_idx]
        except ValueError:
            continue

        valid_permnos = monthly_df[monthly_df['year_month'] == target_ym]['permno'].unique()
        window_daily = daily_ret_df[
            (daily_ret_df['year_month'].isin(past_12_months)) & (daily_ret_df['permno'].isin(valid_permnos))]
        if window_daily.empty: continue
        pivot_daily = window_daily.pivot(index='date', columns='permno', values='ret')
        pivot_daily = pivot_daily[pivot_daily.columns[pivot_daily.count() >= (len(pivot_daily) * MIN_OBS_FRAC)]]
        if pivot_daily.shape[1] < 5: continue
        pivot_daily = pivot_daily.apply(lambda col: col.fillna(pivot_daily.mean(axis=1)), axis=0)

        Y = pivot_daily.values
        n_obs, p_vars = Y.shape
        sample_cov = np.cov(Y, rowvar=False, bias=True)

        meanvar = np.mean(np.diag(sample_cov))
        meancov = (np.sum(sample_cov) - np.sum(np.diag(sample_cov))) / (p_vars * (p_vars - 1))
        target = (meanvar - meancov) * np.eye(p_vars) + meancov * np.ones((p_vars, p_vars))

        # 判断使用固定值还是计算最优值
        if fixed_shrinkage is not None:
            shrinkage = fixed_shrinkage
        else:
            # 原有的 Ledoit-Wolf 计算逻辑
            Y_demeaned = Y - np.mean(Y, axis=0)
            sample2 = (Y_demeaned ** 2).T @ (Y_demeaned ** 2) / n_obs
            pi_hat = np.sum(sample2 - sample_cov ** 2)
            gamma_hat = np.sum((sample_cov - target) ** 2)
            rho_diag = np.sum(np.diag(sample2)) / p_vars - np.sum(np.diag(sample_cov) ** 2) / p_vars
            sum1, sum2 = np.sum(Y_demeaned, axis=1), np.sum(Y_demeaned ** 2, axis=1)
            rho_off = (np.sum((sum1 ** 2 - sum2) ** 2) / p_vars / n_obs - (
                        np.sum(sample_cov) - np.sum(np.diag(sample_cov))) ** 2 / p_vars) / (p_vars - 1)
            shrinkage = max(0, min(1, (pi_hat - (rho_diag + rho_off)) / gamma_hat / n_obs)) if gamma_hat > 0 else 0

        final_cov = (1 - shrinkage) * sample_cov + shrinkage * target
        cov_results.append({
            'year_month': target_ym,
            'valid_stocks': pivot_daily.columns.tolist(),
            'cov_matrix': final_cov * 21,
            'shrinkage_val': shrinkage
        })
    return cov_results


# ==========================================
# 核心4：优化与弹性计算 (保持不变)
# ==========================================
def solve_qp_osqp(cov_matrix, mean_vector, tolerance=OPTIMIZATION_TOL):
    n = len(mean_vector)
    P, q = sparse.csc_matrix(cov_matrix), -mean_vector.astype(np.float64)
    A, l, u = sparse.eye(n, format='csc'), np.zeros(n), np.full(n, np.inf)
    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, eps_abs=1e-14, eps_rel=1e-14, max_iter=100000, verbose=False)
    res = solver.solve()
    if res.info.status != 'solved': return None, None
    weights = res.x
    pos_mask = weights > tolerance
    weights[~pos_mask] = 0.0
    return weights, pos_mask


def compute_benchmark_ur_and_elasticity(cov_results, expected_return_df, ppt_phi):
    results = []
    for cov_item in tqdm(cov_results, desc="逐月优化计算"):
        ym, valid_stocks, cov_matrix = cov_item['year_month'], cov_item['valid_stocks'], cov_item['cov_matrix']
        month_mu = expected_return_df[
            (expected_return_df['year_month'] == ym) & (expected_return_df['permno'].isin(valid_stocks))].set_index(
            'permno').reindex(valid_stocks)
        valid_mask = ~month_mu['expected_return'].isna()
        if not valid_mask.any(): continue
        clean_mu, clean_cov = month_mu.loc[valid_mask, 'expected_return'].values, cov_matrix[
            np.ix_(valid_mask.values, valid_mask.values)]
        weights, pos_mask = solve_qp_osqp(clean_cov, clean_mu)
        if weights is None or np.sum(pos_mask) == 0: continue
        cov_hold = clean_cov[np.ix_(pos_mask, pos_mask)]
        try:
            inv_cov_hold = np.linalg.inv(cov_hold)
        except:
            inv_cov_hold = np.linalg.pinv(cov_hold)
        avg_monthly_ur = np.mean(weights[pos_mask] / np.diag(inv_cov_hold))
        if avg_monthly_ur <= 0: continue
        results.append({'year_month': ym, 'n_hold': np.sum(pos_mask), 'avg_monthly_ur': avg_monthly_ur,
                        'demand_elasticity': 1 + (ppt_phi / avg_monthly_ur), 'shrinkage': cov_item['shrinkage_val']})

    results_df = pd.DataFrame(results)
    results_df = results_df[results_df['n_hold'] >= 10].reset_index(drop=True)
    print("\n" + "=" * 60)
    print("===== Benchmark 基准模型最终结果 =====")
    print(f"全样本平均需求弹性: {results_df['demand_elasticity'].mean():.2f}")
    print(f"全样本平均 Unspanned Return (UR): {results_df['avg_monthly_ur'].mean():.6f}")
    print(f"使用的固定收缩参数 (Shrinkage): {results_df['shrinkage'].mean():.4f}")
    print("=" * 60)
    return results_df


# ==========================================
# 主运行流
# ==========================================


if __name__ == "__main__":
    # 定义你想要跑的所有参数列表
    shrinkage_params = [0.05, 0.1, 0.2, 0.5, 0.75, 0.9, 0.95]

    # 用一个列表存下所有结果的汇总，方便对比
    summary_results = []

    # 预先加载数据（避免在循环里重复加载，节省时间）
    config = {
        'monthly_data_path': 'monthly_stock_returns_with_rf1.csv',
        'monthly_data_dir': '.',
        'daily_data_dir': '.',
        'output_dir': 'benchmark_model_results'
    }
    os.makedirs(config['output_dir'], exist_ok=True)

    monthly_df, daily_ret_df = load_benchmark_data(
        config['monthly_data_path'], config['daily_data_dir'], config['monthly_data_dir']
    )
    ppt_phi, expected_return_df = run_fama_macbeth_for_mu_and_ppt(monthly_df)

    # 开始循环跑不同的收缩参数
    for s_val in shrinkage_params:
        print(f"\n\n>>> 正在运行收缩参数: {s_val} ...")

        # 1. 计算协方差
        cov_results = compute_rolling_covariance(monthly_df, daily_ret_df, fixed_shrinkage=s_val)

        # 2. 计算弹性和 UR
        res_df = compute_benchmark_ur_and_elasticity(cov_results, expected_return_df, ppt_phi)

        # 3. 保存该参数的具体月度结果，文件名区分开
        res_df.to_csv(os.path.join(config['output_dir'], f'results_shrinkage_{s_val}.csv'), index=False)

        # 4. 记录汇总信息
        summary_results.append({
            'Shrinkage': s_val,
            'Avg_UR': res_df['avg_monthly_ur'].mean(),
            'Avg_Elasticity': res_df['demand_elasticity'].mean(),
            'Valid_Months': len(res_df)
        })

    # 最后打印出一张对比表
    summary_df = pd.DataFrame(summary_results)
    print("\n" + "=" * 60)
    print("      所有固定收缩参数测试汇总结果")
    print("=" * 60)
    print(summary_df)
    print("=" * 60)
    summary_df.to_csv(os.path.join(config['output_dir'], 'all_params_summary.csv'), index=False)