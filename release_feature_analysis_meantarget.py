import pandas as pd
from matplotlib import pyplot as plt


def plot_mean_target_trend(df, feature_col, target_col, bins=10, output_path='trend_plot.png'):
    """连续变量分箱后，绘制均值趋势图"""
    df = df.copy()
    df['binned'] = pd.cut(df[feature_col], bins=bins)
    grouped = df.groupby('binned')[target_col].mean().reset_index()
    grouped['bin_mid'] = grouped['binned'].apply(lambda x: x.mid)

    plt.figure(figsize=(8, 5))
    plt.plot(grouped['bin_mid'], grouped[target_col], marker='o', linestyle='-')
    plt.xlabel(feature_col)
    plt.ylabel(f'Mean {target_col}')
    plt.title(f'Mean {target_col} by Binned {feature_col}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"目标均值趋势图已保存至 {output_path}")

df = pd.read_excel("release_data/sup_4_release_data.xlsx")

plot_mean_target_trend(df, feature_col='Polymer_MW', target_col='release_percentage',
                       output_path='release_feature_figures/sup4_trend_Polymer_MW.png')
