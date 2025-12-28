import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams

# ==================== 字体设置代码 ====================
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
# ==========================================================
# --- 全局绘图参数设置 ---
sz=15

rcParams['font.family'] = ['Times New Roman', 'SimSun']
rcParams['axes.titlesize'] = sz+3       # 设置标题字体大小
rcParams['axes.labelsize'] = sz       # 设置x和y轴标签字体大小
rcParams['xtick.labelsize'] = sz-3
rcParams['ytick.labelsize'] = sz-3
rcParams['legend.fontsize'] = sz-2

def data_extraction(df, n):
    """从CV结果中提取指定轮次的测试集数据。"""
    dataframe = pd.DataFrame({
        'Time': df['Time'][n],
        'Experimental_Release': df['Experimental_Release'][n],
        'Predicted_Release': df['Predicted_Release'][n],
        'Experimental Index': df['Experimental Index'][n],
        'DP_Groups': df['DP_Groups'][n]
    })
    return dataframe

if __name__ == '__main__':
    # ============================ 配置区域 ============================
    RAW_DATA_FILEPATH = "release_data/sup_6m_release_data.xlsx"
    # CV_RESULTS_PKL_PATH = "release_best_cv_results/cv_results_Hybrid_Autoregressive_Model_2_shot.pkl"

    CV_RESULTS_PKL_PATH = "release_NESTED_CV_RESULTS_hyb/cv_results_sup_6m_Hybrid_Autoregressive_Model_2_shot.pkl"
    OUTPUT_FOLDER = "release_feature_figures_zh"
    CV_RUN_INDEX = 0

    # <--- 新增: 设置分位数的数量 (例如10代表将数据分为10等份)
    NUM_QUANTILES = 10
    # ================================================================

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- 加载数据 (与之前版本相同) ---
    try:
        df_raw = pd.read_excel(RAW_DATA_FILEPATH)
        df_cv_results = pd.read_pickle(CV_RESULTS_PKL_PATH)
        df_test = data_extraction(df_cv_results, CV_RUN_INDEX)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件！请检查路径。 {e}")
        exit()

    # --- 第一步和第二步分析 (与之前版本相同) ---
    print("--- 1. 时间跨度描述性统计 ---")
    print(df_raw['time'].describe())
    print("-" * 30)
    print("\n--- 2. 各独立实验（sample_id）时长分布分析 ---")
    experiment_durations = df_raw.groupby('sample_id')['time'].max()
    print(experiment_durations.describe())
    # (此处省略直方图的重复代码，您可以根据需要保留)
    print("-" * 30)

    # --- 第三步：模型误差随时间变化分析 (使用分位数) ---
    print(f"\n--- 3. 模型预测误差随时间变化分析 (按{NUM_QUANTILES}等份频数分组) ---")

    # 计算每个预测点的绝对误差
    df_test['Absolute_Error'] = (df_test['Predicted_Release'] - df_test['Experimental_Release']).abs()

    # <--- 修改: 使用pd.qcut进行等频分箱
    # q=NUM_QUANTILES 表示将数据按时间排序后，切成10等份
    # duplicates='drop' 用于处理时间点重复的情况，避免报错
    try:
        df_test['Time_Bin'] = pd.qcut(df_test['Time'], q=NUM_QUANTILES, duplicates='drop')
    except ValueError as e:
        print(f"无法进行等频分箱，可能因为数据点过少或分布极端。错误: {e}")
        exit()

    # 按时间区间计算平均误差
    error_by_time_quantile = df_test.groupby('Time_Bin')['Absolute_Error'].mean().dropna()
    # 计算每个箱子里的数据点数量
    counts_by_time_quantile = df_test.groupby('Time_Bin').size()

    print("按时间分位区间的平均绝对误差 (MAE):")
    # 将频数和平均误差合并展示
    result_df = pd.DataFrame({
        '平均绝对误差': error_by_time_quantile,
        '数据点频数': counts_by_time_quantile
    })
    print(result_df)

    # 绘制误差随时间变化的条形图
    plt.figure(figsize=(8, 5))
    ax = error_by_time_quantile.plot(kind='bar', width=0.8, edgecolor='k', alpha=0.75,
                                     title='MAE随释放时间变化趋势图',
                                     xlabel='释放时间区间',
                                     ylabel='平均绝对误差 (MAE)')

    # 在每个条形图上标注数据点数量
    for i, (count, patch) in enumerate(zip(counts_by_time_quantile, ax.patches)):
        ax.text(patch.get_x() + patch.get_width() / 2., patch.get_height() * 1.01,
                f'n={count}', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    error_plot_path = os.path.join(OUTPUT_FOLDER, 'zh_error_vs_time_quantiles.png')
    plt.savefig(error_plot_path, dpi=600)
    plt.close()
    print(f"\n误差趋势图已保存至: '{error_plot_path}'")

    print(f"\n✅ 时间跨度分析完成！所有结果保存在 '{OUTPUT_FOLDER}' 文件夹中。")