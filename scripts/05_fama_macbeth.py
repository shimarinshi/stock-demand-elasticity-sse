import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import warnings
import os
from tqdm import tqdm

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==========================================
# 全局配置
# ==========================================
MARKETTYPE_SSE = 1  # 上证A股代码
# 定义逐步回归的变量顺序
ORDERED_VARS = [
    'log_prc',
    'log_prc_lag',
    'std_log_vol',
    'std_log_me',
    'monthly_roa',
    'monthly_invest_return',
    'asset_growth',
    'monthly_leverage'
]

# 【关键设置】务必保持为 False。
# 因为基准模型只有6个变量，它没有因为缺少 'asset_growth' 而丢弃样本。
# 若设为True，所有模型都会以最小公共样本运行，模型6的样本量会发生变化从而无法对齐基准。
FORCE_COMMON_SAMPLE = False


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


def descriptive_statistics(data, output_dir='fama_macbeth_results'):
    """
    生成指定变量的描述性统计表格，严格匹配学术论文标准格式
    """
    print("\n" + "=" * 60)
    print("步骤1-补充：生成变量描述性统计")
    print("=" * 60)

    # 严格按照您的要求定义统计变量列表，已剔除log_prc_lag
    stats_vars = [
        'exret',
        'log_prc',
        'std_log_vol',
        'std_log_me',
        'monthly_roa',
        'monthly_invest_return',
        'asset_growth',
        'monthly_leverage'
    ]

    # 自动过滤数据中实际存在的变量，避免因缺失列报错
    available_stats_vars = [var for var in stats_vars if var in data.columns]
    if len(available_stats_vars) == 0:
        print("警告：未找到可统计的有效变量，跳过描述性统计")
        return pd.DataFrame()

    # 初始化统计结果容器
    stats_rows = []

    # 逐变量计算统计指标
    for var in available_stats_vars:
        # 仅用该变量的非缺失值计算，符合学术规范
        series = data[var].dropna()
        if len(series) == 0:
            continue

        # 计算您要求的全部指标
        n = len(series)
        mean = series.mean()
        sd = series.std()
        q5 = series.quantile(0.05)
        q25 = series.quantile(0.25)
        q50 = series.quantile(0.5)
        q75 = series.quantile(0.75)
        q95 = series.quantile(0.95)

        # 格式化输出，和原回归结果格式对齐
        stats_rows.append({
            'Variable': var,
            'N': f"{n:,}",  # 千分位分隔，和原代码Obs格式统一
            'Mean': f"{mean:.4f}",
            'SD': f"{sd:.4f}",
            '5%': f"{q5:.4f}",
            '25%': f"{q25:.4f}",
            '50%': f"{q50:.4f}",
            '75%': f"{q75:.4f}",
            '95%': f"{q95:.4f}"
        })

    # 转为标准DataFrame
    desc_table = pd.DataFrame(stats_rows)

    # 控制台打印结果，和原代码输出风格统一
    print("\n变量描述性统计汇总:")
    print("-" * 80)
    print(desc_table.to_string(index=False))
    print("-" * 80)

    # 自动保存CSV文件，和回归结果放在同一目录
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'descriptive_statistics.csv')
    desc_table.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"描述性统计表格已保存至：{output_csv}")

    return desc_table
# ==========================================
# 核心1：加载数据并进行变量构建
# ==========================================
def load_and_prepare_data(monthly_data_path, monthly_data_dir='.'):
    """
    加载月度收益数据及财务数据，合并并构造所有 Fama-MacBeth 回归所需的自变量
    """
    print("=" * 60)
    print("步骤1：加载上证A股月度数据并合并特征")
    print("=" * 60)

    monthly_df = load_data(monthly_data_path, "月度收益率数据")
    if 'yearmonth' in monthly_df.columns and 'year_month' not in monthly_df.columns:
        monthly_df = monthly_df.rename(columns={'yearmonth': 'year_month'})

    leverage_df = load_data(os.path.join(monthly_data_dir, 'monthly_leverage.csv'), "杠杆率数据")
    roa_investment_df = load_data(os.path.join(monthly_data_dir, 'monthly_roa_investment.csv'), "ROA数据")
    asset_growth_df = load_data(os.path.join(monthly_data_dir, 'monthly_asset_growth.csv'), "资产增长数据")

    # 数据合并
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
    if 'Markettype' in monthly_df.columns:
        monthly_df = monthly_df[monthly_df['Markettype'] == MARKETTYPE_SSE].reset_index(drop=True)
    monthly_df['permno'] = format_permno(monthly_df['permno'])

    # 【核心修复】：严格对齐基准代码的先后顺序！
    # 基准代码是在 load_and_clean_data 中先按照全市场的20%市值门槛剔除，
    # 之后才进入 run_fama_macbeth_for_ppt 剔除无效的价格和成交量数据。

    # 1. 先按月筛选市值前80%
    monthly_df = monthly_df.dropna(subset=['market_equity', 'exret'])

    def filter_by_market_cap(group):
        q20 = group['market_equity'].quantile(0.2)
        return group[group['market_equity'] > q20]

    monthly_df = monthly_df.groupby('year_month').apply(filter_by_market_cap).reset_index(drop=True)

    # 2. 再过滤无效价格、市值和交易量数据
    data = monthly_df.copy()
    data = data[(data['prc'] > 0) & (data['vol'] > 0) & (data['market_equity'] > 0)]

    print("开始构造回归变量...")
    data['log_prc'] = np.log(data['prc'].abs())
    data['log_prc_lag'] = data.groupby('permno')['log_prc'].shift(1)
    data['log_vol'] = np.log(data['vol'])
    data['log_me'] = np.log(data['market_equity'])
    data = data.replace([np.inf, -np.inf], np.nan)

    # 标准化截面变量
    data['std_log_vol'] = standardize_monthly('log_vol', 'year_month', data)
    data['std_log_me'] = standardize_monthly('log_me', 'year_month', data)

    # 将需要的列全部转为数值类型
    for col in ORDERED_VARS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # 下个月的超额收益作为因变量
    data['exret_next'] = data.groupby('permno')['exret'].shift(-1)

    available_vars = [v for v in ORDERED_VARS if v in data.columns]

    if FORCE_COMMON_SAMPLE:
        data = data.dropna(subset=['exret_next'] + available_vars)
        print(f"【统一样本模式】数据清洗完毕，最终有效样本量：{len(data)}")
    else:
        # 新逻辑：不强制删除所有变量缺失，以保证模型6的数据原汁原味对齐基准模型
        base_vars = ['exret_next', 'log_prc', 'log_prc_lag', 'std_log_vol', 'std_log_me']
        base_vars = [v for v in base_vars if v in data.columns]
        data = data.dropna(subset=base_vars)
        print(f"【最大样本模式】数据清洗完毕，基础样本量：{len(data)}")
        print("  (每个模型将独立删除自身所需特征存在的空值缺失行)")

    return data, available_vars


# ==========================================
# 核心2：Fama-MacBeth 逐步回归与表格生成
# ==========================================
def run_fama_macbeth_stepwise(data, ordered_vars):
    """
    逐步加入变量运行 Fama-MacBeth 回归，生成类似于论文 Table 2 的汇总表格
    """
    print("\n" + "=" * 60)
    print("步骤2：Fama-MacBeth 逐步回归计算")
    print("=" * 60)

    months = sorted(data['year_month'].unique())
    table_dict = {}  # 储存用于表格拼接的结果
    phi_values = {}  # 保存每个模型的 log_prc 系数的负值 (phi)

    for i in range(len(ordered_vars)):
        model_vars = ordered_vars[:i + 1]
        model_name = f"({i + 1})"
        print(f"Running Model {model_name} with variables: {model_vars}")

        # 每个模型只剔除自己需要用到的特征存在的NA行
        model_data = data.dropna(subset=['exret_next'] + model_vars).copy()
        print(f"  模型 {model_name} 实际使用样本量: {len(model_data)}")

        monthly_coefs = {v: [] for v in model_vars}
        monthly_r2 = []
        total_obs = 0

        for month in tqdm(months, desc=f"Model {model_name} Iterations", leave=False):
            month_data = model_data[model_data['year_month'] == month]
            if len(month_data) < 10:
                continue

            X = sm.add_constant(month_data[model_vars])
            y = month_data['exret_next']
            try:
                model = sm.OLS(y, X).fit()
                for v in model_vars:
                    monthly_coefs[v].append(model.params[v])
                monthly_r2.append(model.rsquared)
                total_obs += len(month_data)
            except Exception:
                continue

        # 计算 Fama-MacBeth 时序均值及标准误
        T = len(monthly_r2)
        if T == 0:
            print(f"  警告：模型 {model_name} 无有效回归结果！")
            continue

        model_summary = {}
        for v in model_vars:
            coef_mean = np.mean(monthly_coefs[v])
            coef_se = np.std(monthly_coefs[v], ddof=1) / np.sqrt(T)
            t_stat = coef_mean / coef_se

            # 计算 p 值并分配星号
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=T - 1))
            stars = ""
            if p_val < 0.01:
                stars = "***"
            elif p_val < 0.05:
                stars = "**"
            elif p_val < 0.1:
                stars = "*"

            model_summary[v] = f"{coef_mean:.4f}{stars}"
            model_summary[f"{v}_se"] = f"({coef_se:.4f})"

            # 保存 log_prc 的负系数计算 phi
            if v == 'log_prc':
                phi_values[model_name] = -coef_mean
                print(f"  -> 模型 {model_name} 的 PPT(phi) = {phi_values[model_name]:.6f}")

        model_summary['R2'] = f"{np.mean(monthly_r2) * 100:.2f}%"
        model_summary['Obs'] = f"{total_obs:,}"

        table_dict[model_name] = model_summary

    # 打印 phi 对比
    print("\n" + "-" * 60)
    print("各模型的 PPT(phi) 汇总:")
    for m, p in phi_values.items():
        print(f"  模型 {m}: phi = {p:.6f}")
    print("-" * 60)

    return _format_to_table(table_dict, ordered_vars)


def _format_to_table(table_dict, ordered_vars):
    """
    将结果格式化为类似原论文 Table 2 的 DataFrame
    """
    rows = []

    for var in ordered_vars:
        # 存系数和显著性
        coef_row = [var]
        # 存标准误
        se_row = [""]

        for i in range(len(ordered_vars)):
            col = f"({i + 1})"
            if col in table_dict and var in table_dict[col]:
                coef_row.append(table_dict[col][var])
                se_row.append(table_dict[col][f"{var}_se"])
            else:
                coef_row.append("")
                se_row.append("")
        rows.append(coef_row)
        rows.append(se_row)

    # 添加 Obs 和 R^2 行
    obs_row = ["Obs"]
    r2_row = ["R2"]
    for i in range(len(ordered_vars)):
        col = f"({i + 1})"
        if col in table_dict:
            obs_row.append(table_dict[col]['Obs'])
            r2_row.append(table_dict[col]['R2'])
        else:
            obs_row.append("")
            r2_row.append("")

    rows.append(obs_row)
    rows.append(r2_row)

    # 构造 DataFrame
    columns = ["Dependent variable: next-month return"] + [f"({i + 1})" for i in range(len(ordered_vars))]
    final_table = pd.DataFrame(rows, columns=columns)

    return final_table


# ==========================================
# 主运行流
# ==========================================
def main_fama_macbeth_table2():
    print("=" * 80)
    print("Fama-MacBeth 逐步回归管线启动 (复现 Table 2 格式)")
    if FORCE_COMMON_SAMPLE:
        print("模式：所有模型使用统一样本（会为满足后续变量删减基础模型样本量）")
    else:
        print("模式：每个模型使用最大样本（模型6将精确对齐您的基准模型结果）")
    print("=" * 80)

    config = {
        'monthly_data_path': 'monthly_stock_returns_with_rf1.csv',
        'monthly_data_dir': '.',
        'output_root_dir': 'fama_macbeth_results'
    }

    os.makedirs(config['output_root_dir'], exist_ok=True)

    try:
        # 1. 加载并清理数据
        data, available_vars = load_and_prepare_data(config['monthly_data_path'], config['monthly_data_dir'])

        if data.empty:
            raise ValueError("清洗后数据集为空，请检查基础数据及其合并条件。")
        desc_table = descriptive_statistics(data, config['output_root_dir'])
        # 2. 运行回归并生成表格
        final_table = run_fama_macbeth_stepwise(data, available_vars)

        # 3. 输出展示与落盘
        print("\n\n" + "=" * 80)
        print("Fama-MacBeth 逐步回归结果汇总 (Table 2 Format):")
        print("Note: *p<0.1; **p<0.05; ***p<0.01. Standard errors in parentheses.")
        print("-" * 80)
        print(final_table.to_string(index=False))
        print("=" * 80)

        output_csv = os.path.join(config['output_root_dir'], 'table2_fama_macbeth_stepwise.csv')
        final_table.to_csv(output_csv, index=False)
        print(f"回归结果表格已成功保存至：{output_csv}")

    except Exception as e:
        print(f"程序执行失败：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_fama_macbeth_table2()