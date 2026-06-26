import pandas as pd
import numpy as np

# 读取CSV数据
df = pd.read_csv('assetgrowth.csv', dtype={'Stkcd': str})

# 筛选Typrep为'A'的数据（假设使用合并报表）
df = df[df['Typrep'] == 'A'].copy()

# 仅删除关键字段的缺失值（替代全局dropna，避免误删有效数据）
df = df.dropna(subset=['F080501A']).copy()

# 转换日期和数值列
df['Accper'] = pd.to_datetime(df['Accper'])
df['F080501A'] = pd.to_numeric(df['F080501A'], errors='coerce')  # 资产增长率

# 按股票代码分组处理
monthly_data = []
# 新增：存储原始锚点信息，用于验证
anchor_verify = []

for stkcd, group in df.groupby('Stkcd'):
    # 按日期排序（加copy避免SettingWithCopyWarning）
    group = group.sort_values('Accper').copy()

    # 获取公司名称（无则用股票代码，和之前逻辑统一）
    company_name = group['ShortName'].iloc[0] if 'ShortName' in group.columns else stkcd

    # 确保数据点足够进行插值（至少2个）
    if len(group) < 2:
        print(f"警告: 股票 {stkcd} ({company_name}) 数据点不足，跳过")
        continue

    # 新增：保存原始锚点信息（用于验证，和之前逻辑统一）
    anchor_info = group[['Accper', 'F080501A']].reset_index(drop=True)
    anchor_info['permno'] = stkcd
    anchor_info['ShortName'] = company_name
    anchor_verify.append(anchor_info)

    # 计算资产指数：假设期初资产指数为1（保留原逻辑，优化NaN处理）
    asset_index = [1.0]  # 起始值
    for r in group['F080501A']:
        if not np.isnan(r):
            new_index = asset_index[-1] * (1 + r)
        else:
            # 若增长率缺失，保持上期值（原逻辑保留）
            new_index = asset_index[-1]
        asset_index.append(new_index)

    # 调整索引长度（去掉起始的1，匹配报告期）
    asset_index = asset_index[1:]
    group['asset_index'] = asset_index

    # 设置日期索引
    group.set_index('Accper', inplace=True)

    # 生成月度日期范围（每月最后一天，和之前逻辑统一）
    min_date = group.index.min()
    max_date = group.index.max()
    month_ends = pd.date_range(start=min_date, end=max_date, freq='M')

    # ========== 核心统一：先插值原始指数（time方法），再对数 ==========
    # 1. 对原始资产指数做时间加权插值（替代原先对数后线性插值）
    asset_index_series = group['asset_index'].reindex(month_ends).interpolate(method='time')

    # 2. 插值后再取对数（避免对数线性插值导致的月度值恒定）
    log_series = np.log(asset_index_series)

    # 3. 计算月度对数变化，转换为增长率（原逻辑保留）
    log_change = log_series.diff()
    monthly_growth = np.exp(log_change) - 1

    # 构建结果DataFrame（新增ShortName，和之前格式统一）
    result = pd.DataFrame({
        'permno': stkcd,
        'ShortName': company_name,
        'date': month_ends,
        'asset_growth': monthly_growth,
        # 可选：保留插值后的原始指数，方便验证
        'asset_index_interpolated': asset_index_series.values
    })
    monthly_data.append(result)

# 检查是否有数据被处理（和之前逻辑统一）
if not monthly_data:
    print("错误: 没有足够的数据进行处理")
    exit()

# 合并所有股票的数据
monthly_df = pd.concat(monthly_data, ignore_index=True)

# 将日期转换为年月格式的整数（和之前逻辑统一）
monthly_df['year_month'] = monthly_df['date'].dt.year * 100 + monthly_df['date'].dt.month

# 按permno和date排序
monthly_df = monthly_df.sort_values(['permno', 'date']).reset_index(drop=True)

# 显示处理后的数据（完善输出，和之前格式统一）
print("前20行数据:")
print(monthly_df.head(20))
print(f"\n数据形状: {monthly_df.shape}")
print(f"\n时间范围: {monthly_df['date'].min()} 到 {monthly_df['date'].max()}")
print(f"\n唯一股票数量: {monthly_df['permno'].nunique()}")
print(f"\n每个股票的平均月度观测值: {len(monthly_df) / monthly_df['permno'].nunique():.1f}")

# 检查缺失值（新增，和之前逻辑统一）
print(f"\n缺失值统计:")
print(
    f"asset_growth缺失: {monthly_df['asset_growth'].isna().sum()} ({monthly_df['asset_growth'].isna().sum() / len(monthly_df) * 100:.1f}%)")

# 新增：验证原始锚点值（确认未被修改）
print("\n原始锚点值验证（前10行）:")
anchor_df = pd.concat(anchor_verify, ignore_index=True)
print(anchor_df.head(10))

# 保存到CSV文件
monthly_df.to_csv('monthly_asset_growth.csv', index=False)
print(f"\n数据已保存到 monthly_asset_growth.csv")

# 可选：显示月度资产增长率统计（和之前逻辑统一）
print("\n月度资产增长率统计:")
print(monthly_df['asset_growth'].describe())