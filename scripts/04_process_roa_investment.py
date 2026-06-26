import pandas as pd
import numpy as np

# 读取CSV数据
df = pd.read_csv('ROA.csv', dtype={'Stkcd': str})

# 筛选Typrep为'A'的数据（假设使用合并报表）
df = df[df['Typrep'] == 'A'].copy()

# 删除关键字段的缺失值
df = df.dropna(subset=['F050201B', 'F053202B']).copy()

# 转换日期和数值列
df['Accper'] = pd.to_datetime(df['Accper'])
df['F050201B'] = pd.to_numeric(df['F050201B'], errors='coerce')  # ROA
df['F053202B'] = pd.to_numeric(df['F053202B'], errors='coerce')  # 投资收益率

# 按股票代码分组处理
monthly_data = []

for stkcd, group in df.groupby('Stkcd'):
    # 按日期排序
    group = group.sort_values('Accper').copy()

    # 获取公司名称（取第一个）
    company_name = group['ShortName'].iloc[0] if 'ShortName' in group.columns else stkcd

    # 确保数据点足够进行插值
    if len(group) < 2:
        print(f"警告: 股票 {stkcd} ({company_name}) 数据点不足，跳过")
        continue

    # 为ROA计算资产指数：假设期初资产指数为1
    roa_index = [1.0]  # 起始值
    for r in group['F050201B']:
        new_index = roa_index[-1] * (1 + r)
        roa_index.append(new_index)

    # 为投资收益率计算收益指数：假设期初收益指数为1
    invest_index = [1.0]  # 起始值
    for r in group['F053202B']:
        new_index = invest_index[-1] * (1 + r)
        invest_index.append(new_index)

    # 调整索引长度
    roa_index = roa_index[1:]  # 去掉起始的1，使长度与报告期一致
    invest_index = invest_index[1:]  # 去掉起始的1，使长度与报告期一致

    group['roa_index'] = roa_index
    group['invest_index'] = invest_index

    # 设置日期索引（确保是datetime类型）
    group = group.set_index('Accper')

    # 生成月度日期范围（每月最后一天，使用BM避免非交易日）
    min_date = group.index.min()
    max_date = group.index.max()
    # 生成所有月度最后一天，确保覆盖整个时间范围
    month_ends = pd.date_range(start=min_date, end=max_date, freq='M')

    # ========== 核心修改部分 ==========
    # 1. 先对原始指数进行时间序列插值（而非对数后插值）
    # 使用method='time'更适合时间序列插值
    roa_index_series = group['roa_index'].reindex(month_ends).interpolate(method='time')
    invest_index_series = group['invest_index'].reindex(month_ends).interpolate(method='time')

    # 2. 插值后再取对数
    log_roa_series = np.log(roa_index_series)
    log_invest_series = np.log(invest_index_series)

    # 3. 计算月度对数变化，转换为增长率
    log_roa_change = log_roa_series.diff()
    monthly_roa_growth = np.exp(log_roa_change) - 1

    log_invest_change = log_invest_series.diff()
    monthly_invest_growth = np.exp(log_invest_change) - 1
    # ========== 核心修改结束 ==========

    # 构建结果DataFrame
    result = pd.DataFrame({
        'permno': stkcd,
        'ShortName': company_name,
        'date': month_ends,
        'monthly_roa': monthly_roa_growth,
        'monthly_invest_return': monthly_invest_growth,
        # 可选：保留插值后的原始指数，方便验证
        'roa_index_interpolated': roa_index_series.values,
        'invest_index_interpolated': invest_index_series.values
    })
    monthly_data.append(result)

# 检查是否有数据被处理
if not monthly_data:
    print("错误: 没有足够的数据进行处理")
    exit()

# 合并所有股票的数据
monthly_df = pd.concat(monthly_data, ignore_index=True)

# 将日期转换为年月格式的整数（如199901）
monthly_df['year_month'] = monthly_df['date'].dt.year * 100 + monthly_df['date'].dt.month

# 按permno和date排序
monthly_df = monthly_df.sort_values(['permno', 'date']).reset_index(drop=True)

# 显示处理后的数据
print("前20行数据:")
print(monthly_df.head(20))
print(f"\n数据形状: {monthly_df.shape}")
print(f"\n时间范围: {monthly_df['date'].min()} 到 {monthly_df['date'].max()}")
print(f"\n唯一股票数量: {monthly_df['permno'].nunique()}")
print(f"\n每个股票的平均月度观测值: {len(monthly_df) / monthly_df['permno'].nunique():.1f}")

# 检查缺失值
print(f"\n缺失值统计:")
print(
    f"monthly_roa缺失: {monthly_df['monthly_roa'].isna().sum()} ({monthly_df['monthly_roa'].isna().sum() / len(monthly_df) * 100:.1f}%)")
print(
    f"monthly_invest_return缺失: {monthly_df['monthly_invest_return'].isna().sum()} ({monthly_df['monthly_invest_return'].isna().sum() / len(monthly_df) * 100:.1f}%)")

# 保存到CSV文件（可选）
monthly_df.to_csv('monthly_roa_investment.csv', index=False)
print(f"\n数据已保存到 monthly_roa_investment.csv")

# 可选：显示一些统计信息
print("\n月度ROA增长率统计:")
print(monthly_df['monthly_roa'].describe())

print("\n月度投资收益率增长率统计:")
print(monthly_df['monthly_invest_return'].describe())