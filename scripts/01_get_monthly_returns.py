# ---------------------- 1. 导入所需库 ----------------------
import pandas as pd
import numpy as np

# ---------------------- 2. 配置文件路径与列名（请根据你的实际文件修改） ----------------------
# 股票月度数据文件路径
stock_file_path = "stockprice1.csv"
# 无风险利率日度数据文件路径
rf_file_path = "noriskrate.csv"
# 最终输出文件路径
output_file_path = "monthly_stock_returns_with_rf1.csv"

# ====================== 核心修改1：无风险利率文件列配置 ======================
# 【重要】根据你的截图，文件有表头（如Clsdt），请直接填写列名
# 如果你不确定列名，可以先运行代码，看打印出的原始列名是什么
rf_date_col = "Clsdt"  # 日期列的列名（对应你截图中的第一列）
rf_value_col = 2        # 无风险利率列的索引（第三列，python从0开始）；如果知道列名建议填列名字符串
# ====================================================================

# ---------------------- 3. 读取并处理无风险利率数据 ----------------------
# ====================== 核心修改2：正确读取有表头的文件 ======================
# 把 header=None 改为 header=0，明确告诉pandas第一行是列名
# 编码保留 utf-8-sig，避免中文乱码
rf_df = pd.read_csv(rf_file_path, header=0, encoding="utf-8-sig")

# 可选：打印原始列名，方便你确认配置是否正确
print("===== 无风险利率文件原始列名 =====")
print(rf_df.columns.tolist())
print("\n")

# ====================== 核心修改3：安全稳健的日期处理逻辑 ======================
# 步骤1：强制把日期列转为字符串，清洗掉可能的空值
rf_df[rf_date_col] = rf_df[rf_date_col].astype(str).str.strip()

# 步骤2：彻底过滤掉混入数据的表头行（防止文件中间有重复表头）
rf_df = rf_df[rf_df[rf_date_col] != rf_date_col]

# 步骤3：用pandas智能解析日期（兼容 1994/2/1、1994-02-01、19940201 等几乎所有常见格式）
# errors="coerce" 表示把无法解析的无效日期转为空值，后续会自动过滤
rf_df["parsed_date"] = pd.to_datetime(rf_df[rf_date_col], errors="coerce")

# 步骤4：过滤掉日期解析失败的行
rf_df = rf_df.dropna(subset=["parsed_date"])

# 步骤5：生成标准的 YYYYMM 格式年月列（如 199402），完美转为int
# 彻底替代原来容易出错的 str[:6] 切片方式
rf_df["yearmonth"] = rf_df["parsed_date"].dt.strftime("%Y%m").astype(int)

# 重命名无风险利率列（兼容列索引和列名两种配置）
if isinstance(rf_value_col, int):
    # 如果配置的是列索引
    rf_df = rf_df.rename(columns={rf_df.columns[rf_value_col]: "rf"})
else:
    # 如果配置的是列名
    rf_df = rf_df.rename(columns={rf_value_col: "rf"})

# 去重：每个月只保留1个无风险利率值
rf_monthly_df = rf_df.groupby("yearmonth", as_index=False)["rf"].first()

# 校验：查看是否有重复月份
print(f"无风险利率数据处理完成，共覆盖{len(rf_monthly_df)}个月份")
print("无风险利率前5行预览：")
print(rf_monthly_df.head())
print("\n")

# ---------------------- 4. 读取并处理股票月度数据 ----------------------
# 读取股票数据文件
stock_df = pd.read_csv(stock_file_path, encoding="utf-8-sig")

# --------------- 4.1 日期格式处理（同样优化为更稳健的方式） ---------------
if pd.api.types.is_datetime64_any_dtype(stock_df["Trdmnt"]):
    # 若为datetime格式，直接格式化
    stock_df["date"] = stock_df["Trdmnt"].dt.strftime("%Y%m").astype(int)
else:
    # 若为字符串/数字，先用to_datetime解析，再格式化（避免切片错误）
    stock_df["parsed_trdmnt"] = pd.to_datetime(stock_df["Trdmnt"], errors="coerce")
    stock_df = stock_df.dropna(subset=["parsed_trdmnt"])
    stock_df["date"] = stock_df["parsed_trdmnt"].dt.strftime("%Y%m").astype(int)

# yearmonth列和date列值一致
stock_df["yearmonth"] = stock_df["date"]

# --------------- 4.2 计算滞后一期市值（lag_market_equity） ---------------
stock_df = stock_df.sort_values(by=["Stkcd", "date"]).reset_index(drop=True)
stock_df["lag_market_equity"] = stock_df.groupby("Stkcd")["Msmvttl"].shift(1)

# --------------- 4.3 字段重命名 ---------------
stock_rename_map = {
    "Stkcd": "permno",
    "Mretwd": "ret",
    "Mretnd": "retx",
    "Mnshrtrd": "vol",
    "Msmvttl": "market_equity",
    "Mclsprc": "prc"
}
stock_df = stock_df.rename(columns=stock_rename_map)

# 处理prc列
if "prc" not in stock_df.columns:
    stock_df["prc"] = np.nan
    print("提示：当前股票数据中无prc(月收盘价)字段，已生成空列")

# ---------------------- 5. 合并股票数据与无风险利率 ----------------------
final_df = pd.merge(
    left=stock_df,
    right=rf_monthly_df,
    on="yearmonth",
    how="left"
)

# 计算超额收益
final_df["exret"] = final_df["ret"] - final_df["rf"]

# ---------------------- 6. 调整列顺序 ----------------------
final_columns_order = [
    "permno", "date", "prc", "ret", "retx", "vol",
    "yearmonth", "market_equity", "rf", "exret",
    "lag_market_equity", "Markettype"
]

# 安全检查：防止某些列不存在导致报错
existing_columns = [col for col in final_columns_order if col in final_df.columns]
missing_columns = [col for col in final_columns_order if col not in final_df.columns]
if missing_columns:
    print(f"\n警告：以下列不存在，将跳过：{missing_columns}")

final_df = final_df[existing_columns]

# ---------------------- 7. 导出最终文件 ----------------------
final_df.to_csv(output_file_path, index=False, encoding="utf-8")
print(f"\n文件生成完成！已保存至：{output_file_path}")
print("最终数据前5行预览：")
print(final_df.head())
