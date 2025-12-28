import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==================== 字体设置代码 ====================
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
# ==========================================================

# --- 全局绘图参数设置 ---
sz=15

rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = sz+3       # 设置标题字体大小
rcParams['axes.titleweight'] = 'bold' # 设置标题加粗
rcParams['axes.labelsize'] = sz       # 设置x和y轴标签字体大小
rcParams['axes.labelweight'] = 'bold' # 设置x和y轴标签加粗
rcParams['xtick.labelsize'] = sz-3
rcParams['ytick.labelsize'] = sz-3
rcParams['legend.fontsize'] = sz-2

if __name__ == '__main__':
    # ============================ 配置区域 ============================
    DATA_FILE_PATH = "release_data/sup_6m_release_data.xlsx" # <--- 请确保这是您的数据文件路径

    # 定义您想要分析的特征列
    STATIC_FEATURES = ['Polymer_MW', 'LA/GA', 'Polymer_size', 'Zeta_Potential', 'PDI', 'EE',
                       'Drug_MW', 'LogP', 'HBD', 'HBA', 'HA', 'RB', 'PSA', 'NAR', 'FSP3']

    OUTPUT_FOLDER = "release_feature_figures"
    OUTPUT_FIGURE_FILENAME = "cumulative_explained_variance_plot.png"
    # ================================================================

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- 1. 加载并准备数据 ---
    print("--- 步骤1: 加载并准备数据 ---")
    try:
        df = pd.read_excel(DATA_FILE_PATH)
        # 每个sample_id的静态特征都是重复的，我们只需要一份
        df_unique_samples = df.drop_duplicates(subset=['sample_id'])
        X = df_unique_samples[STATIC_FEATURES]
        print(f"成功加载并提取了 {len(X)} 个独立样本的 {len(STATIC_FEATURES)} 个静态特征。")
    except Exception as e:
        print(f"数据加载过程中出错: {e}"); exit()

    # --- 2. 特征标准化 ---
    # PCA对数据的尺度非常敏感，因此标准化是必须的步骤
    print("\n--- 步骤2: 对特征进行标准化 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("特征标准化完成。")

    # --- 3. 执行PCA ---
    print("\n--- 步骤3: 执行主成分分析 (PCA) ---")
    # 设置n_components=None，让PCA计算出所有可能的主成分
    pca = PCA(n_components=None)
    pca.fit(X_scaled)

    # --- 4. 计算累积解释方差 ---
    # explained_variance_ratio_ 包含了每个主成分能解释的方差比例
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    print(f"各主成分解释的方差比例: \n{explained_variance_ratio}")
    print(f"\n累积解释方差: \n{cumulative_explained_variance}")

    # --- 5. 绘图 ---
    print("\n--- 步骤5: 绘制累积解释方差图 ---")
    plt.figure(figsize=(8, 5))

    num_components = len(explained_variance_ratio)
    component_indices = np.arange(1, num_components + 1)

    # 绘制每个主成分的解释方差条形图
    plt.bar(component_indices, explained_variance_ratio, alpha=0.5, align='center',
            label='Variance explained by a single principal component')

    # 绘制累积解释方差折线图
    plt.plot(component_indices, cumulative_explained_variance,
             label='Cumulative explained variance', color='red', marker='o', linestyle='-')

    # 找到达到99.99%阈值需要的主成分数量并标注
    n_components_9999 = np.where(cumulative_explained_variance >= 0.9999)[0][0] + 1
    n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
    n_components_90 = np.where(cumulative_explained_variance >= 0.90)[0][0] + 1

# 绘制90%和95%的参考线
    plt.axhline(y=0.9999, color='purple', linestyle='--', label=f'99.99% {n_components_9999} Principal Components')
    plt.axhline(y=0.95,  color='g', linestyle='--', label=f'95% {n_components_95} Principal Components')
    plt.axhline(y=0.90,  color='orange', linestyle='--', label=f'90% {n_components_90} Principal Components')


    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Number of Principal Components')
    plt.title('Cumulative Explained Variance (PCA)')
    plt.xticks(component_indices)
    plt.ylim(0, 1.05)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FIGURE_FILENAME)
    plt.savefig(output_path, dpi=600)
    plt.close()

    print(f"\n✅ 分析完成！图表已保存至: '{output_path}'")
    print(f"分析结论：需要 {n_components_9999} 个主成分来捕捉数据中99.99%的信息。")