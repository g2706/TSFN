import pandas as pd
import seaborn as sns
import os
import numpy as np
from matplotlib import pyplot as plt, rcParams
from matplotlib.lines import Line2D
from matplotlib import ticker

# ==================== 字体设置代码 ====================
# plt.rcParams['font.sans-serif'] = ['SimHei']
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

def plot_binned_violin_with_mean_axis(df, feature_col, target_col, output_path, num_bins=5):
    """
    先对连续特征进行分箱，绘制小提琴图，然后手动将x轴标签替换为各分箱的均值。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_plot = df.copy().dropna(subset=[feature_col, target_col])

    # --- 1. 分箱并计算各箱均值 ---
    bin_labels = [f'Bin {i+1}' for i in range(num_bins)]
    try:
        df_plot[f'{feature_col}_Binned'] = pd.cut(df_plot[feature_col], bins=num_bins, labels=bin_labels)
    except ValueError as e:
        print(f"错误：无法将特征 '{feature_col}' 分成 {num_bins} 个箱子。 {e}")
        return

    # 计算每个分箱的 feature_col (EE) 的实际均值，用于后续的标签替换
    x_tick_labels_means = df_plot.groupby(f'{feature_col}_Binned', observed=True)[feature_col].mean()
    print("每个分箱的EE均值（将作为横轴标签）:")
    print(x_tick_labels_means)

    # --- 2. 绘图主体 ---
    plt.figure(figsize=(8, 5))

    # <--- 核心修正1: x轴使用分类的'Binned'列，让seaborn能正确绘制5个等距的提琴
    ax = sns.violinplot(data=df_plot, x=f'{feature_col}_Binned', y=target_col,
                        hue=f'{feature_col}_Binned',  # <--- 新增
                        inner='box',
                        palette='pastel',
                        legend=False)              # <--- 新增

    # 计算每个小提琴上release_percentage的均值
    means_y = df_plot.groupby(f'{feature_col}_Binned', observed=True)[target_col].mean()

    # 绘制均值点 (菱形标记)
    # 此时x轴的位置是整数 0, 1, 2, 3, 4
    ax.scatter(x=np.arange(len(means_y)), y=means_y.values,
               marker='D', color='black', s=50, zorder=3)

    # <--- 核心修正2: 手动设置横轴的刻度标签 ---
    # get_xticks() 会得到 [0, 1, 2, 3, 4]
    # 我们用计算好的EE均值来作为这些位置的显示文本
    ax.set_xticks(ax.get_xticks()) # 确保获取到刻度位置
    ax.set_xticklabels([f'{mean:.2f}' for mean in x_tick_labels_means.values])
    # --------------------------------------------------------

    plt.xlabel(f'Mean {feature_col} Value')
    plt.ylabel('Release Percentage')
    plt.title(f'Distribution of Release Under Different {feature_col}')

    # # --- 创建图例 ---
    # legend_elements = [
    #     Line2D([0], [0], marker='D', color='w', label='均值 (Mean Release %)',
    #            markerfacecolor='black', markersize=10)
    # ]
    # ax.legend(handles=legend_elements, title="图例")

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"\n已修正坐标轴的小提琴图已保存至 {output_path}")

# --- 主程序 ---
try:
    df = pd.read_excel("release_data/sup_6m_release_data.xlsx")

    plot_binned_violin_with_mean_axis(
        df,
        feature_col='EE',
        target_col='release_percentage',
        output_path='release_feature_figures/sup6_violin_EE_mean_axis_final_FIXED.png',
        num_bins=5
    )
except FileNotFoundError:
    print("错误：找不到数据文件。请检查文件路径。")
except Exception as e:
    print(f"发生未知错误: {e}")