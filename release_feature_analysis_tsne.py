# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
#
#
# def plot_tsne(X_scaled, y, output_path='tsne_plot.png'):
#     tsne = TSNE(n_components=2, random_state=42, perplexity=50)
#     X_embedded = tsne.fit_transform(X_scaled)
#
#     plt.figure(figsize=(8, 6))
#     sc = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)
#     plt.colorbar(sc, label='release_percentage')
#     plt.title('t-SNE Projection')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=600)
#     plt.close()
#     print(f"t-SNE 可视化图已保存至 {output_path}")
#
# df = pd.read_excel("release_data/sup_6_release_data.xlsx")
# X = df.drop(columns=['sample_id', 'drug_name', 'source', 'release_percentage'])
#
# stdScale = StandardScaler().fit(X)
# X_scale = stdScale.transform(X)
#
# plot_tsne(X_scale, df['release_percentage'], output_path='release_feature_figures/sup6_tsne.png')




import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- 设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("release_data/sup_6_release_data.xlsx")

# --- 修改: 按 sample_id 聚合数据 ---
# 使用 drop_duplicates 来确保每个 sample_id 只保留一行
df_agg = df.drop_duplicates(subset=['sample_id']).reset_index(drop=True)

# 现在 X_agg 的每一行都代表一个独立的配方
X_agg = df_agg.drop(columns=['sample_id', 'drug_name', 'source', 'release_percentage', 'time'])

# 我们可以用 drug_name 来为点着色，看看不同药物是否会分开
# 将药物名称转换为数字类别用于着色
y_color_labels, drug_map = pd.factorize(df_agg['drug_name'])

# --- 后续流程不变 ---
stdScale = StandardScaler().fit(X_agg)
X_scale = stdScale.transform(X_agg)

tsne = TSNE(n_components=2, random_state=42, perplexity=30) # Perplexity 建议设为 5-50 之间
X_embedded = tsne.fit_transform(X_scale)

# --- 绘图 ---
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_color_labels, cmap='tab20', s=50, alpha=0.8)
plt.title('基于配方静态特征的t-SNE聚类', fontsize=16)

# 创建一个图例来显示药物名称
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=drug,
                              markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
                   for i, drug in enumerate(drug_map)]
plt.legend(handles=legend_elements, title="药物名称", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.85, 1]) # 为图例留出空间
output_path = 'release_feature_figures/sup6_tsne_by_sample.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"按配方聚合的t-SNE可视化图已保存至 {output_path}")