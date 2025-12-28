import pandas as pd
import umap
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


def plot_umap(X_scaled, y, output_path='umap_plot.png'):
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='plasma', s=30, alpha=0.7)
    plt.colorbar(sc, label='release_percentage')
    plt.title('UMAP Projection')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"UMAP 可视化图已保存至 {output_path}")

df = pd.read_excel("release_data/sup_6_release_data.xlsx")
X = df.drop(columns=['sample_id', 'drug_name', 'source', 'release_percentage'])

stdScale = StandardScaler().fit(X)
X_scale = stdScale.transform(X)

plot_umap(X_scale, df['release_percentage'], output_path='release_feature_figures/sup6_umap.png')
