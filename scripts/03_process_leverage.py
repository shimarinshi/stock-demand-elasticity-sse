import pandas as pd
import numpy as np

# 读取CSV数据
df = pd.read_csv('leverage.csv', dtype={'Stkcd': str})

# 筛选Typrep为'A'的数据（假设使用合并报表）
df = df[df['Typrep'] == 'A'].copy()

# 删除关键字段的缺失值
df = df.dropna(subset=['F070301B']).copy()

# 转换日期和数值列
df['Accper'] = pd.to_datetime(df['Accper'])
df['F070301B'] = pd.to_numeric(df['F070301B'], errors='coerce')  # 杠杆率

# 按股票代码分组处理
monthly_data = []
# 新增：存储锚点信息，方便验证
anchor_verify = []

for stkcd, group in df.groupby('Stkcd'):
    # 按日期排序（加copy避免SettingWithCopyWarning）
    group = group.sort_values('Accper').copy()

    # 获取公司名称（取第一个）
    company_name = group['ShortName'].iloc[0] if 'ShortName' in group.columns else stkcd

    # 确保数据点足够进行插值
    if len(group) < 2:
        print(f"警告: 股票 {stkcd} ({company_name}) 数据点不足，跳过")
        continue

    # 新增：保存原始锚点信息（用于验证）
    anchor_info = group[['F070301B']].reset_index()
    anchor_info['permno'] = stkcd
    anchor_info['ShortName'] = company_name
    anchor_verify.append(anchor_info)

    # 设置日期索引
    group.set_index('Accper', inplace=True)

    # 生成月度日期范围（所有月底，用M或BM，BM更贴合交易日）
    min_date = group.index.min()
    max_date = group.index.max()
    month_ends = pd.date_range(start=min_date, end=max_date, freq='M')  # 每月最后一天

    # ========== 核心优化：改用time插值 ==========
    # 杠杆率是存量指标，直接插值合理；用time替代linear，按实际天数插值
    leverage_series = group['F070301B'].reindex(month_ends).interpolate(method='time')

    # 构建结果DataFrame
    result = pd.DataFrame({
        'permno': stkcd,
        'ShortName': company_name,
        'date': month_ends,
        'monthly_leverage': leverage_series
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
    f"monthly_leverage缺失: {monthly_df['monthly_leverage'].isna().sum()} ({monthly_df['monthly_leverage'].isna().sum() / len(monthly_df) * 100:.1f}%)")

# 新增：验证原始锚点值（确认未被修改）
print("\n原始锚点值验证（前10行）:")
anchor_df = pd.concat(anchor_verify, ignore_index=True)
print(anchor_df.head(10))

# 保存到CSV文件（可选）
monthly_df.to_csv('monthly_leverage.csv', index=False)
print(f"\n数据已保存到 monthly_leverage.csv")

# 可选：显示一些统计信息
print("\n月度杠杆率统计:")
print(monthly_df['monthly_leverage'].describe())

# 可选：查看各股票数据情况
print("\n各股票数据统计:")
stock_stats = monthly_df.groupby('permno').agg({
    'ShortName': 'first',
    'date': ['min', 'max', 'count'],
    'monthly_leverage': 'mean'
}).round(4)

# 重命名列
stock_stats.columns = ['ShortName', 'start_date', 'end_date', 'monthly_obs', 'avg_leverage']
print(stock_stats.sort_values('start_date').head(20))