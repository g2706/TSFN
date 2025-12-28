import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, rcParams
import matplotlib.cm as cm
import matplotlib.colors as colors
import pickle

# ==================== 1. 字体和绘图设置 ====================
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
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

# ==================== 2. 核心绘图函数 ====================
def plot_feature_contribution_and_performance(importance_df, performance_df, top_n_highlight=7, bar_gamma=0.5):
    """
    绘制特征贡献度与模型性能的组合图。
    """
    sorted_features = importance_df['feature']
    importances = importance_df['importance']
    mean_scores = performance_df['mean_score']
    std_scores = performance_df['std_score']

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel('Features')
    ax1.set_ylabel('Feature Importance')

    cmap_bar = plt.get_cmap('Blues')
    norm_bar = colors.PowerNorm(gamma=bar_gamma, vmin=0, vmax=importances.max())
    bar_colors = cmap_bar(norm_bar(importances))
    ax1.bar(sorted_features, importances, color=bar_colors, alpha=0.8)
    ax1.tick_params(axis='y')
    plt.xticks(rotation=90)

    for i, tick in enumerate(ax1.get_xticklabels()):
        if i < top_n_highlight:
            tick.set_color('red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean R²')
    cmap_line = plt.get_cmap('Reds')
    norm_line = colors.Normalize(vmin=0, vmax=len(sorted_features))
    for i in range(len(sorted_features) - 1):
        segment_color = cmap_line(norm_line(len(sorted_features) - i))
        ax2.plot(sorted_features[i:i+2], mean_scores[i:i+2], color=segment_color, linestyle='-', linewidth=1.5)
        ax2.fill_between(sorted_features[i:i+2], (mean_scores - std_scores)[i:i+2], (mean_scores + std_scores)[i:i+2], color=segment_color, alpha=0.2)
    ax2.scatter(sorted_features, mean_scores, c=np.arange(len(sorted_features))[::-1], cmap=cmap_line, norm=norm_line, edgecolor='black', s=30, zorder=3)
    ax2.tick_params(axis='y')
    ax2.set_ylim(bottom=max(0, mean_scores.min() - 0.1), top=min(1, mean_scores.max() + 0.1))

    plt.title(f'Feature Contribution and R² Performance Curve')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig

# ==================== 3. 主执行流程 ====================
if __name__ == '__main__':
    # ============================ 配置区域 ============================
    # <--- 1. 在这里指定由脚本一生成的.pkl数据文件
    DATA_FILE_PATH = "release_feature_figures/feature_analysis_data.pkl"

    # <--- 2. 您可以在这里自由调整绘图参数
    TOP_N_HIGHLIGHT = 7
    BAR_COLOR_GAMMA = 0.5 # 控制条形图颜色过渡速度，越小越平缓
    OUTPUT_FILENAME = "release_feature_figures/feature_performance_plot_from_saved_data.png"
    # ================================================================

    # --- 加载已分析好的数据 ---
    print(f"--- 正在从 {DATA_FILE_PATH} 加载已分析好的数据 ---")
    try:
        with open(DATA_FILE_PATH, 'rb') as f:
            analysis_results = pickle.load(f)
        importance_df = analysis_results['importance_df']
        performance_df = analysis_results['performance_df']
        print("数据加载成功！")
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 '{DATA_FILE_PATH}'。请先运行分析脚本。")
        exit()
    except Exception as e:
        print(f"加载数据时出错: {e}")
        exit()

    # --- 生成图表 ---
    print("\n--- 正在生成图表 ---")
    fig = plot_feature_contribution_and_performance(
        importance_df,
        performance_df,
        top_n_highlight=TOP_N_HIGHLIGHT,
        bar_gamma=BAR_COLOR_GAMMA
    )

    fig.savefig(OUTPUT_FILENAME, dpi=600)
    print(f"\n✅ 分析完成！图表已保存至: '{OUTPUT_FILENAME}'")