import pandas as pd
import seaborn as sns
import os
import numpy as np
from matplotlib import pyplot as plt, ticker
from matplotlib.lines import Line2D

# ==================== 字体设置代码 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ==========================================================

def plot_binned_violin_with_mean_axis(df, feature_col, target_col, output_path, num_bins=5):
    """
    先对连续特征进行分箱，然后以每个分箱的均值为横坐标绘制小提琴图。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_plot = df.copy().dropna(subset=[feature_col, target_col])

    # --- 核心修改 ---
    # 1. 仍然先进行分箱，以确定分组
    #    我们只需要分箱的“标签”，所以不需要retbins=True
    bin_labels = [f'Bin {i+1}' for i in range(num_bins)]
    try:
        df_plot[f'{feature_col}_Binned'] = pd.cut(df_plot[feature_col], bins=num_bins, labels=bin_labels)
    except ValueError as e:
        print(f"错误：无法将特征 '{feature_col}' 分成 {num_bins} 个箱子。 {e}")
        return

    # 2. 计算每个箱子内 feature_col 的实际均值
    means_by_bin = df_plot.groupby(f'{feature_col}_Binned', observed=True)[feature_col].mean()
    print("每个分箱的EE均值（将作为横轴坐标）:")
    print(means_by_bin)

    # 3. 创建一个新的列，将每个数据点映射到其所在分箱的均值
    #    这个新列将作为我们绘图的x轴
    df_plot['x_coord'] = df_plot[f'{feature_col}_Binned'].map(means_by_bin)

    # -----------------------------------------

    plt.figure(figsize=(12, 7))

    # <--- 修改: 使用新的 'x_coord' 列作为x轴
    # hue参数现在可以用来区分不同的小提琴
    ax = sns.violinplot(data=df_plot, x='x_coord', y=target_col,
                        hue=f'{feature_col}_Binned', # 使用原始分箱标签来着色
                        inner='box',
                        palette='pastel',
                        legend=False) # 我们将手动创建更清晰的图例

    # 计算并叠加每个小提琴的真实均值点 (与上一版相同)
    means_y = df_plot.groupby('x_coord', observed=True)[target_col].mean()
    ax.scatter(x=np.arange(len(means_y)), y=means_y.values,
               marker='D', color='black', s=50, zorder=3)
    # <--- 核心修改: 格式化横轴刻度标签 ---
    # 1. 创建一个格式化器，指定格式为保留两位小数的浮点数 ('%.2f')
    formatter = ticker.FormatStrFormatter('%.2f')
    # 2. 将此格式化器应用到图表的x轴上
    # ax.xaxis.set_major_formatter(formatter)
    # ----------------------------------------
    plt.xlabel(f'平均 {feature_col} 值', fontsize=12)
    plt.ylabel(target_col, fontsize=12)
    plt.title(f'{target_col} 在不同 {feature_col} 均值分组下的分布', fontsize=16)

    # --- 创建自定义图例 ---
    # 获取每个小提琴的颜色
    palette = sns.color_palette('pastel', n_colors=num_bins)
    legend_elements = [
        # 为均值点创建一个图例项
        Line2D([0], [0], marker='D', color='w', label='均值 (Mean)',
               markerfacecolor='black', markersize=10)
    ]
    # 为每个小提琴创建一个图例项
    for i, label in enumerate(means_by_bin.index):
        mean_val = means_by_bin.iloc[i]
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', label=f'EE均值 ≈ {mean_val:.1f}',
                   markerfacecolor=palette[i], markersize=15)
        )
    ax.legend(handles=legend_elements, title="图例", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 为图例留出空间
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"\n以均值为横坐标的小提琴图已保存至 {output_path}")

# --- 主程序 ---
try:
    df = pd.read_excel("release_data/sup_6m_release_data.xlsx")

    plot_binned_violin_with_mean_axis(
        df,
        feature_col='EE',
        target_col='release_percentage',
        output_path='release_feature_figures/sup6_violin_EE_mean_axis.png',
        num_bins=5
    )
except FileNotFoundError:
    print("错误：找不到数据文件。请检查文件路径。")
except Exception as e:
    print(f"发生未知错误: {e}")