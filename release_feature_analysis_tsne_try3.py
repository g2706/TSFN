import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- 设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================ 配置区域 ============================
DATA_FILE_PATH = "release_data/sup_6_release_data.xlsx"
# 选择一个静态特征来为点着色，以探索其与曲线形状的关系
# 常用选项: 'drug_name', 'LA/GA', 'Polymer_MW'
COLOR_BY_FEATURE = 'Polymer_size'
OUTPUT_FOLDER = "release_shape_figures"
OUTPUT_FIGURE_FILENAME = f"sup6_tsne_by_curve_shape_colored_by_{COLOR_BY_FEATURE}.png"
# ================================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 1. 加载数据 ---
print("--- 步骤1: 加载数据 ---")
df = pd.read_excel(DATA_FILE_PATH)
print("数据加载成功。")

# --- 2. 数据重塑：将长格式转为宽格式 ---
print("\n--- 步骤2: 重塑数据，将每条曲线转换为一个向量 ---")
# 使用pivot_table将time列转换为列标题
# 每一行现在代表一个独立的 sample_id，每一列代表一个时间点
df_wide = df.pivot_table(index='sample_id', columns='time', values='release_percentage', aggfunc='mean')

# --- 3. 处理因时间点不一致导致的缺失值 (NaN) ---
# 使用线性插值填充中间的缺失值，然后用后面的值填充开头的缺失值
df_wide = df_wide.interpolate(method='linear', axis=1).bfill(axis=1)
# 填充后可能还有NaN（如果整行都是NaN），直接丢弃这些不完整的样本
df_wide.dropna(inplace=True)

# X_shapes 的每一行都是一条完整的释放曲线向量
X_shapes = df_wide.values
print(f"数据重塑完成，得到 {X_shapes.shape[0]} 条可用于分析的完整曲线。")

# --- 4. 准备用于着色的标签 ---
# 我们需要将原始的静态特征与重塑后的数据对齐
df_static = df.drop_duplicates(subset=['sample_id']).set_index('sample_id')
df_aligned_static = df_static.loc[df_wide.index] # 确保顺序和ID都与df_wide一致

# 将我们选择的特征列转换为用于着色的数字标签
y_color_labels, category_map = pd.factorize(df_aligned_static[COLOR_BY_FEATURE])

# --- 5. 标准化、t-SNE降维与绘图 ---
print("\n--- 步骤5: 执行t-SNE降维与绘图 ---")
# 对曲线向量进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_shapes)

# 执行t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_embedded = tsne.fit_transform(X_scaled)

# --- 绘图 ---
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_color_labels, cmap='tab20', s=50, alpha=0.8)
plt.title(f'基于“曲线形状”的t-SNE聚类 (按“{COLOR_BY_FEATURE}”着色)', fontsize=16)
plt.xlabel("t-SNE 维度 1")
plt.ylabel("t-SNE 维度 2")
plt.grid(True, linestyle='--', alpha=0.6)

# 创建一个图例来显示类别名称
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                              markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
                   for i, cat in enumerate(category_map)]
plt.legend(handles=legend_elements, title=COLOR_BY_FEATURE, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.85, 1]) # 为图例留出空间
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FIGURE_FILENAME)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ 分析完成！基于曲线形状的t-SNE图已保存至 '{output_path}'")