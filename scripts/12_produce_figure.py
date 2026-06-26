import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ====================== 1. 中文字体+全局字号基础设置 ======================
sys_fonts = {f.name: f for f in font_manager.fontManager.ttflist}
target_font = 'SimHei' if 'SimHei' in sys_fonts else 'Microsoft YaHei'
plt.rcParams['font.sans-serif'] = [target_font]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 24  # <--- 全局基础字号适配26号主字体（从20改为24）

# ====================== 2. 构造数据 ======================
data = {
    'market': ['上证A股', '上证A股', '上证A股', '上证A股', '上证A股',
               'NYSE', 'NYSE', 'NYSE', 'NYSE', 'NYSE'],
    'model': ['CAPM', 'FF3', 'FF5', 'FF6', 'Benchmark',
              'CAPM', 'FF3', 'FF5', 'FF6', 'Benchmark'],
    'x_pos': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'elasticity': [3279.98, 39.54, 32.96, 30.93, 7.77,
                   1701.3, 308.1, 29.9, 34.0, 5]
}
df = pd.DataFrame(data)

cn_df = df[df['market'] == '上证A股'].reset_index(drop=True)
us_df = df[df['market'] == 'NYSE'].reset_index(drop=True)

# ====================== 3. 绘图 ======================
fig, ax = plt.subplots(figsize=(16, 10), dpi=100)

# --- 1. 绘制线条+数据点（Benchmark加原点） ---
# 【上证A股（红色）】
# 实线：CAPM→FF3→FF5→FF6
ax.plot(cn_df[cn_df['model'].isin(['CAPM', 'FF3', 'FF5', 'FF6'])]['x_pos'],
        cn_df[cn_df['model'].isin(['CAPM', 'FF3', 'FF5', 'FF6'])]['elasticity'],
        color='#d62728', linewidth=3, linestyle='-',
        marker='o', markersize=12, markeredgecolor='black', markeredgewidth=1.5, label='上证A股')
# 虚线：FF6→Benchmark（<--- 新增marker='o'，确保Benchmark有原点）
ax.plot(cn_df[cn_df['model'].isin(['FF6', 'Benchmark'])]['x_pos'],
        cn_df[cn_df['model'].isin(['FF6', 'Benchmark'])]['elasticity'],
        color='#d62728', linewidth=2.5, linestyle='--',
        marker='o', markersize=12, markeredgecolor='black', markeredgewidth=1.5)

# 【NYSE美股（蓝色）】
# 实线：CAPM→FF3→FF5→FF6
ax.plot(us_df[us_df['model'].isin(['CAPM', 'FF3', 'FF5', 'FF6'])]['x_pos'],
        us_df[us_df['model'].isin(['CAPM', 'FF3', 'FF5', 'FF6'])]['elasticity'],
        color='#1f77b4', linewidth=3, linestyle='-',
        marker='o', markersize=12, markeredgecolor='black', markeredgewidth=1.5, label='NYSE')
# 虚线：FF6→Benchmark（<--- 新增marker='o'，确保Benchmark有原点）
ax.plot(us_df[us_df['model'].isin(['FF6', 'Benchmark'])]['x_pos'],
        us_df[us_df['model'].isin(['FF6', 'Benchmark'])]['elasticity'],
        color='#1f77b4', linewidth=2.5, linestyle='--',
        marker='o', markersize=12, markeredgecolor='black', markeredgewidth=1.5)

# --- 2. 模型名称标注（字号统一26号） ---
# 上证A股标签（红色，字号26）
for idx, row in cn_df.iterrows():
    if row['model'] == 'Benchmark':
        x_offset = 0.12
        y_offset = 1.04
    else:
        x_offset = 0.1
        y_offset = 1.09
    ax.text(row['x_pos'] + x_offset, row['elasticity'] * y_offset,
            row['model'], fontsize=22, fontweight='bold', color='#d62728') # <--- 标注字号统一26号

# NYSE标签（蓝色，字号26）
for idx, row in us_df.iterrows():
    if row['model'] == 'Benchmark':
        x_offset = 0.12
        y_offset = 0.92
    elif row['model'] in ['FF5', 'FF6']:
        x_offset = 0.1
        y_offset = 0.80
    else:
        x_offset = 0.1
        y_offset = 1.09
    ax.text(row['x_pos'] + x_offset, row['elasticity'] * y_offset,
            row['model'], fontsize=22, fontweight='bold', color='#1f77b4') # <--- 标注字号统一26号

# --- 3. 坐标轴设置（刻度字号统一26号） ---
ax.set_yscale('log')
# 坐标轴标题保持30号不变
ax.set_ylabel('Elasticity (log scale)', fontsize=30, fontweight='bold', labelpad=6)
ax.set_xlabel('Model Specification', fontsize=30, fontweight='bold', labelpad=6)

# X轴：刻度字号统一26号
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['CAPM', 'FF3', 'FF5', 'FF6', 'Benchmark'], fontsize=26) # <--- 刻度字号统一26号
ax.set_xlim(0.7, 6.2)

# Y轴：刻度字号统一26号
ax.set_yticks([5, 10, 50, 100, 500, 1000, 4000])
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
ax.tick_params(axis='y', labelsize=26) # <--- 刻度字号统一26号
ax.set_ylim(3, 4800)

# --- 4. 图例（字号统一26号） ---
ax.legend(loc='upper right', fontsize=26, framealpha=1, edgecolor='black',
          borderpad=1.8, labelspacing=1.5) # <--- 图例字号统一26号

# 网格美化
ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='lightgray')
ax.grid(False, axis='x')
plt.tight_layout()

# ====================== 4. 保存图片 ======================
plt.savefig('需求弹性中美对比图_含FF6_26号字体版_Benchmark带点.png', dpi=300, bbox_inches='tight')
print("图片已保存！字号统一26号，Benchmark已添加原点")
plt.show()