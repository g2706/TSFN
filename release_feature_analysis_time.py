import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ==================== 字体设置代码 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# ==========================================================

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
    # 原始数据文件，用于分析整体时间和时长
    RAW_DATA_FILEPATH = "release_data/sup_6m_release_data.xlsx"
    # CV预测结果文件，用于分析模型误差
    # CV_RESULTS_PKL_PATH = "release_best_cv_results/cv_results_Hybrid_Autoregressive_Model_2_shot.pkl"

    CV_RESULTS_PKL_PATH = "release_NESTED_CV_RESULTS_hyb/cv_results_sup_6m_Hybrid_Autoregressive_Model_2_shot.pkl"
    
    OUTPUT_FOLDER = "release_feature_figures"
    CV_RUN_INDEX = 0 # 分析表现最好的一轮CV结果
    # ================================================================

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- 加载数据 ---
    try:
        df_raw = pd.read_excel(RAW_DATA_FILEPATH)
        df_cv_results = pd.read_pickle(CV_RESULTS_PKL_PATH)
        df_test = data_extraction(df_cv_results, CV_RUN_INDEX)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件！请检查路径。 {e}")
        exit()

    # --- 第一步：描述性统计分析 ---
    print("--- 1. 时间跨度描述性统计 ---")
    print("所有数据点的时间分布情况:")
    print(df_raw['time'].describe())
    print("-" * 30)

    # --- 第二步：实验时长分布分析 ---
    print("\n--- 2. 各独立实验（sample_id）时长分布分析 ---")
    # 计算每个sample_id的最大时间点，即实验时长
    experiment_durations = df_raw.groupby('sample_id')['time'].max()

    print("各实验时长的统计:")
    print(experiment_durations.describe())

    # 绘制时长分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(experiment_durations, bins=30, edgecolor='k', alpha=0.7)
    plt.title('实验时长分布直方图', fontsize=16)
    plt.xlabel('实验时长 (小时)', fontsize=14)
    plt.ylabel('样本数量 (频数)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    duration_plot_path = os.path.join(OUTPUT_FOLDER, 'experiment_duration_histogram.png')
    plt.savefig(duration_plot_path, dpi=300)
    plt.close()
    print(f"时长分布图已保存至: '{duration_plot_path}'")
    print("-" * 30)

    # --- 第三步：模型误差随时间变化分析 ---
    print("\n--- 3. 模型预测误差随时间变化分析 ---")

    # 计算每个预测点的绝对误差
    df_test['Absolute_Error'] = (df_test['Predicted_Release'] - df_test['Experimental_Release']).abs()

    # 创建时间区间 (分箱)
    # 我们可以根据您数据的时间范围来动态创建箱子
    max_time = df_test['Time'].max()
    # 例如，如果最大时间是1500小时，我们就创建 0-100, 100-200, ... 的区间
    bins = np.arange(0, max_time + 100, 100)
    df_test['Time_Bin'] = pd.cut(df_test['Time'], bins=bins, right=False)

    # 按时间区间计算平均误差
    error_by_time = df_test.groupby('Time_Bin')['Absolute_Error'].mean().dropna()
    print("按时间区间的平均绝对误差 (MAE):")
    print(error_by_time)

    # 绘制误差随时间变化的条形图
    plt.figure(figsize=(12, 7))
    error_by_time.plot(kind='bar', width=0.8, edgecolor='k', alpha=0.75)
    plt.title('模型绝对误差随时间的变化趋势', fontsize=16)
    plt.xlabel('时间区间 (小时)', fontsize=14)
    plt.ylabel('平均绝对误差 (MAE)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    error_plot_path = os.path.join(OUTPUT_FOLDER, 'error_vs_time.png')
    plt.savefig(error_plot_path, dpi=300)
    plt.close()
    print(f"误差趋势图已保存至: '{error_plot_path}'")

    print(f"\n✅ 时间跨度分析完成！所有结果保存在 '{OUTPUT_FOLDER}' 文件夹中。")