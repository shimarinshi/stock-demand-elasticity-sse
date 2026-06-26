# 价格传导、资产替代性与股票需求弹性——基于上证A股市场的实证研究

毕业论文的代码与结果。研究上证A股市场中价格传导系数（PPT φ）与资产替代性对股票需求弹性的影响，比较 CAPM、Fama-French 三因子/五因子/六因子模型与 Benchmark 特征模型的需求弹性估计。

## 论文

论文 PDF 位于 [`thesis/`](thesis/) 目录。

## 项目结构

```
stock-demand-elasticity-sse/
├── README.md
├── LICENSE                    # CC BY-ND 4.0
├── .gitignore
├── thesis/
│   └── 价格传导、资产替代性与股票需求弹性——基于上证A股市场的实证研究.pdf
├── scripts/
│   ├── 01_get_monthly_returns.py        # 计算月度超额收益率
│   ├── 02_process_asset_growth.py       # 处理资产增长率
│   ├── 03_process_leverage.py           # 处理杠杆率
│   ├── 04_process_roa_investment.py     # 处理 ROA 与投资收益率
│   ├── 05_fama_macbeth.py              # Fama-MacBeth 描述性统计（论文 Table 2）
│   ├── 06_benchmark_model.py           # Benchmark 模型（全 shrinkage 水平）
│   ├── 07_classic_factor_models.py     # CAPM / FF3 / FF5 / FF6
│   ├── 08_benchmark_industry.py        # Benchmark 行业异质性
│   ├── 09_benchmark_value.py           # Benchmark 市值分组异质性
│   ├── 10_classic_factor_industry.py   # 经典因子模型行业异质性
│   ├── 11_classic_factor_value.py      # 经典因子模型市值分组异质性
│   ├── 12_produce_figure.py            # 生成论文 Table 1 比较图
│   └── benchmark_model_basic.py        # 参考：单 shrinkage 水平 Benchmark
└── results/
    ├── benchmark_model_results/
    ├── classic_factor_models_results/
    ├── classic_factor_models_results_industry/
    └── fama_macbeth_results/
```

## 数据说明

本项目使用的原始数据来自 **CSMAR（国泰安）数据库**，因版权限制**无法在本仓库中分发**。复现研究需自行从 CSMAR 下载以下数据。

### 所需 CSMAR 数据表

所有数据文件需放在脚本同级目录下运行。

#### 1. 个股日度收益率

- **CSMAR 表**：股票日个股回报率文件 (TRD_Dalyr)
- **预期文件名**：`*dailyret*.csv`（如 `0004dailyret.csv`、`1419dailyret1.csv` 等）
- **关键列**：
  | 列名 | 含义 | 备注 |
  |------|------|------|
  | Stkcd | 股票代码 | 脚本内统一格式化为 6 位 permno |
  | Trddt | 交易日期 | 格式 YYYY-MM-DD |
  | Dretwd | 考虑现金红利再投资的日个股回报率 | |
  | Markettype | 市场类型 | 脚本自动过滤 Markettype=1（上证A股） |
- **时间范围**：1994 年 1 月 - 2025 年 12 月，按年份分段导出为多个 CSV
- **使用者**：`06_benchmark_model.py`、`08_benchmark_industry.py`、`09_benchmark_value.py`

#### 2. 个股月度数据

- **CSMAR 表**：股票月个股回报率文件 (TRD_Mnth)
- **预期文件名**：`stockprice1.csv`
- **关键列**：
  | 列名 | 含义 |
  |------|------|
  | Stkcd / permno | 股票代码 |
  | Markettype | 市场类型（=1 上证A股） |
  | yearmonth | 年月（YYYYMM） |
  | prc | 月末收盘价 |
  | vol | 月交易量（股） |
  | market_equity | 个股流通市值 |
- **使用者**：`01_get_monthly_returns.py` → 生成 `monthly_stock_returns_with_rf1.csv`

#### 3. 日度无风险利率

- **CSMAR 表**：无风险利率文件 (TRD_Nrrate)
- **预期文件名**：`noriskrate.csv`
- **关键列**：
  | 列名 | 含义 |
  |------|------|
  | Clsdt | 日期 |
  | 第 3 列（数值列） | 日度无风险利率 |
- **使用者**：`01_get_monthly_returns.py`

#### 4. 资产增长率

- **CSMAR 表**：资产负债表 (FS_Combas)
- **预期文件名**：`assetgrowth.csv`
- **关键列**：
  | 列名 | 含义 | 备注 |
  |------|------|------|
  | Stkcd | 股票代码 | |
  | Typrep | 报表类型 | 脚本过滤 Typrep='A'（合并报表） |
  | Accper | 会计期间截止日 | 脚本转为 datetime |
  | F080501A | 总资产增长率 | 用于计算月度 asset_growth |
  | ShortName | 公司简称 | 日志输出用 |
- **使用者**：`02_process_asset_growth.py` → 生成 `monthly_asset_growth.csv`

#### 5. 杠杆率

- **CSMAR 表**：资产负债表 (FS_Combas)
- **预期文件名**：`leverage.csv`
- **关键列**：
  | 列名 | 含义 | 备注 |
  |------|------|------|
  | Stkcd | 股票代码 | |
  | Typrep | 报表类型 | 过滤 Typrep='A' |
  | Accper | 会计期间截止日 | |
  | F070301B | 杠杆率 | 用于计算月度 monthly_leverage |
  | ShortName | 公司简称 | |
- **使用者**：`03_process_leverage.py` → 生成 `monthly_leverage.csv`

#### 6. ROA 与投资收益率

- **CSMAR 表**：利润表 (FS_Comins)
- **预期文件名**：`ROA.csv`
- **关键列**：
  | 列名 | 含义 | 备注 |
  |------|------|------|
  | Stkcd | 股票代码 | |
  | Typrep | 报表类型 | 过滤 Typrep='A' |
  | Accper | 会计期间截止日 | |
  | F050201B | 总资产净利润率 (ROA) | |
  | F053202B | 投资收益率 | 用于 monthly_invest_return |
- **使用者**：`04_process_roa_investment.py` → 生成 `monthly_roa_investment.csv`

#### 7. Fama-French 五因子

- **CSMAR 表**：三因子/五因子模型因子数据
- **预期文件名**：`fivefactor.csv`
- **使用者**：`07_classic_factor_models.py`、`10_classic_factor_industry.py`、`11_classic_factor_value.py`

#### 8. 动量因子

- **CSMAR 表**：动量因子数据
- **预期文件名**：`momfactor.csv`
- **使用者**：`07_classic_factor_models.py`、`10_classic_factor_industry.py`、`11_classic_factor_value.py`

#### 9. 行业分类

- **CSMAR 表**：行业分类（证监会 2012 版）
- **预期文件名**：`industry_num.csv`
- **使用者**：`08_benchmark_industry.py`、`10_classic_factor_industry.py`

## 复现步骤

### 阶段一：数据预处理（生成中间文件）

按顺序运行，每个脚本生成一个中间 CSV：

| 序号 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 1 | `01_get_monthly_returns.py` | `stockprice1.csv` + `noriskrate.csv` | `monthly_stock_returns_with_rf1.csv` |
| 2 | `02_process_asset_growth.py` | `assetgrowth.csv` | `monthly_asset_growth.csv` |
| 3 | `03_process_leverage.py` | `leverage.csv` | `monthly_leverage.csv` |
| 4 | `04_process_roa_investment.py` | `ROA.csv` | `monthly_roa_investment.csv` |

### 阶段二：核心分析

脚本 01-04 的输出需与原始 CSMAR 文件放在同一目录：

| 序号 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 5 | `05_fama_macbeth.py` | 4 个中间 CSV | `results/fama_macbeth_results/` |
| 6 | `06_benchmark_model.py` | 4 个中间 CSV + `*dailyret*.csv` | `results/benchmark_model_results/` |
| 7 | `07_classic_factor_models.py` | 4 个中间 CSV + `fivefactor.csv` + `momfactor.csv` | `results/classic_factor_models_results/` |

### 阶段三：异质性分析

| 序号 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 8 | `08_benchmark_industry.py` | 4 个中间 CSV + `*dailyret*.csv` + `industry_num.csv` | `results/benchmark_model_results/` |
| 9 | `09_benchmark_value.py` | 4 个中间 CSV + `*dailyret*.csv` | `results/benchmark_model_results/` |
| 10 | `10_classic_factor_industry.py` | 4 个中间 CSV + 因子 + `industry_num.csv` | `results/classic_factor_models_results_industry/` |
| 11 | `11_classic_factor_value.py` | 4 个中间 CSV + 因子 | `results/classic_factor_models_results/`（FF6 value 子目录） |

### 阶段四：制图

| 序号 | 脚本 | 说明 |
|------|------|------|
| 12 | `12_produce_figure.py` | 生成论文 Table 1（中美市场弹性对比图），数据已硬编码 |

### 参考脚本

`benchmark_model_basic.py` 是 Benchmark 模型的单 shrinkage 水平（φ=0.1）演示版本，用于理解模型结构。

## 依赖

项目使用 Python 3.12 开发，主要依赖：

- `pandas` `numpy` — 数据处理
- `statsmodels` — Fama-MacBeth 回归
- `scikit-learn` — 标准化
- `scipy` — 稀疏矩阵与优化
- `osqp` — 二次规划求解（**Windows 需 osqp>=0.6.5**）
- `matplotlib` — 制图
- `tqdm` — 进度条

> 本仓库不提供 `requirements.txt`，因版本兼容性问题（尤其是 `osqp` 的 C 求解器依赖）建议按需 `pip install`。如遇 `osqp` 安装失败，请参考 [OSQP 官方安装指南](https://osqp.org/)，或使用 conda：`conda install -c conda-forge osqp`。

## 结果文件

`results/` 目录包含本论文的完整实证结果：

- **`benchmark_model_results/`** — Benchmark 模型弹性结果（含 shrinkage 0.05/0.1/0.2/0.5/0.75/0.9/0.95 各水平）、行业与市值异质性
- **`classic_factor_models_results/`** — CAPM、FF3、FF5、FF6 弹性结果及因子 beta
- **`classic_factor_models_results_industry/`** — 经典因子模型行业异质性
- **`fama_macbeth_results/`** — 描述性统计与逐步回归结果

## 引用

若使用本项目代码或结果，请引用论文：

> 张伯一. 价格传导、资产替代性对股票需求弹性的影响研究——基于上证A股市场的实证检验[D]. 2026.

## 许可证

本项目（代码、结果、论文）采用 [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/) 许可证。您可以自由分享，但必须署名且不得修改后分发。
