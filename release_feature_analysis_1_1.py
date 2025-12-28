import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr
import seaborn as sns

# ==================== 字体和绘图设置 ====================
# 我们将字体设置得更通用一些，以确保中文能正确显示
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
# =======================================================
# --- 全局绘图参数设置 ---
sz=18

rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = sz+3       # 设置标题字体大小
rcParams['axes.titleweight'] = 'bold' # 设置标题加粗
rcParams['axes.labelsize'] = sz       # 设置x和y轴标签字体大小
rcParams['axes.labelweight'] = 'bold' # 设置x和y轴标签加粗
rcParams['xtick.labelsize'] = sz-3
rcParams['ytick.labelsize'] = sz-3
rcParams['legend.fontsize'] = sz-2

# --- 1. 数据加载与准备 (与您的代码相同) ---
# 读取你转换后的 Excel 文件
df = pd.read_excel("release_data/sup_6m_release_data.xlsx")

# 只保留数值型特征列
X = df.drop(columns=['sample_id', 'drug_name', 'source', 'release_percentage'])

# 计算 Spearman 相关矩阵
corr = spearmanr(X).correlation
corr = (corr + corr.T) / 2  # 保证对称
np.fill_diagonal(corr, 1)

# --- 2. 使用 seaborn.clustermap 一步完成绘图 ---
print("--- 正在生成聚类热图 ---")

# 创建 images 目录（如果不存在）
os.makedirs("release_feature_figures_ultra", exist_ok=True)

# 调用 clustermap 函数
# 它会自动完成：计算距离、执行层次聚类、重排矩阵、绘制热图和树状图
g = sns.clustermap(
    corr,                   # 输入您的相关性矩阵
    method='ward',          # 使用与您之前相同的 'ward' 聚类方法
    cmap='RdBu_r',          # 使用与您之前相同的红蓝反转色板
    vmin=-1, vmax=1,        # 设置颜色范围
    figsize=(8, 8),       # 设置图形大小
    linewidths=.5,          # 在热图单元格之间添加细线
    xticklabels=X.columns,  # 设置x轴和y轴的标签
    yticklabels=X.columns
)

# --- 3. 调整和美化图表 ---
# 旋转x轴标签以便阅读
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
# 确保y轴标签完全可见
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

# 添加一个总标题
g.fig.suptitle('Spearman Correlation & Ward Linkage',
               fontsize=sz+3, weight='bold', y=1.02) # y=1.02 将标题向上移动一点

# 保存图像
output_path = "release_feature_figures_ultra/sup_6m_combined_clustermap.png"
g.savefig(output_path, dpi=600, bbox_inches='tight')

plt.close() # 关闭图像释放内存

print(f"✅ 聚类热图已成功保存至: '{output_path}'")