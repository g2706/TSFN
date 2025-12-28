import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy

# 读取你转换后的 Excel 文件
df = pd.read_excel("release_data/sup_6m_release_data.xlsx")

# 只保留数值型特征列（排除样本编号、药物名、文献名、time、release_percentage）
# X = df.drop(columns=['sample_id', 'drug_name', 'source', 'release_percentage'])

X = df.drop(columns=['sample_id', 'drug_name', 'source', 'release_percentage',
                     ])

# # 可选：移除重复行（有些特征行是重复的）
# X = X.drop_duplicates()

# 计算 Spearman 相关矩阵
corr = spearmanr(X).correlation
corr = (corr + corr.T) / 2  # 保证对称
np.fill_diagonal(corr, 1)

# 将相关性转换为距离矩阵（用于层次聚类）
distance_matrix = 1 - np.abs(corr)

# # 强制对称化，消除数值误差
# distance_matrix = (distance_matrix + distance_matrix.T) / 2
#
# # 对角线设为 0
# np.fill_diagonal(distance_matrix, 0)

dist_linkage = hierarchy.ward(squareform(distance_matrix))

# 创建 images 目录（如果不存在）
os.makedirs("release_feature_figures_ultra", exist_ok=True)

# 层次聚类树状图
fig1, ax1 = plt.subplots(figsize=(12, 8))
dendro = hierarchy.dendrogram(dist_linkage, labels=X.columns.tolist(), ax=ax1, leaf_rotation=90)
ax1.set_title("Hierarchical Clustering (Ward-linkage)", fontsize=14, weight="bold")
ax1.set_xlabel('FEATURE NAMES', fontsize=14)
ax1.set_ylabel('HEIGHT', fontsize=14)
ax1.tick_params(axis='both', labelsize=12)
fig1.tight_layout()
fig1.savefig("release_feature_figures_ultra/sup_6m_hierarchical_clustering.png", dpi=300, bbox_inches='tight')
plt.close(fig1)  # 关闭图像释放内存

# Spearman 相关性热图
dendro_idx = np.arange(0, len(dendro["ivl"]))
fig2, ax2 = plt.subplots(figsize=(6, 8))
# 使用深色代表强相关，浅色代表弱相关
im = ax2.imshow(
    corr[dendro["leaves"], :][:, dendro["leaves"]],
    cmap='RdBu_r',  # 可换为 'seismic', 'bwr' 等
    vmin=-1, vmax=1
)

ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
fig2.colorbar(im, format='%.2f')
ax2.tick_params(axis='both', labelsize=12)
ax2.set_title("Spearman's Rank Correlation", fontsize=14, weight="bold")
fig2.tight_layout()
fig2.savefig("release_feature_figures_ultra/sup_6m_spearman_correlation.png", dpi=300, bbox_inches='tight')
plt.close(fig2)

