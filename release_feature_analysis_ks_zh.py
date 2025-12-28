import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams
# <--- 新增: 导入K-S检验函数
from scipy.stats import ks_2samp

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

def plot_ks_curve(data1, data2, name1='Data 1', name2='Data 2', output_path='ks_plot.png'):
    """
    计算并绘制两个数据集的K-S曲线图。
    """
    # 1. 计算K-S统计量和p值
    ks_statistic, p_value = ks_2samp(data1, data2)

    # 2. 准备绘制CDF所需的数据
    # 对数据进行排序
    data1_sorted = np.sort(data1)
    data2_sorted = np.sort(data2)

    # 计算每个点的累积概率
    # y轴代表 "小于等于当前x值的样本所占的比例"
    cdf1 = np.arange(1, len(data1_sorted) + 1) / len(data1_sorted)
    cdf2 = np.arange(1, len(data2_sorted) + 1) / len(data2_sorted)

    # 3. 绘图
    plt.figure(figsize=(8, 5))

    # 使用阶梯图(step plot)来精确表示经验CDF
    plt.step(data1_sorted, cdf1, label=f'{name1} 累积分布函数', where='post')
    plt.step(data2_sorted, cdf2, label=f'{name2} 累积分布函数', where='post')

    # 4. 找到并标记最大差异点
    # 合并所有x值以找到评估差异的最佳位置
    all_values = np.unique(np.concatenate([data1_sorted, data2_sorted]))
    cdf1_interp = np.interp(all_values, data1_sorted, cdf1, right=1)
    cdf2_interp = np.interp(all_values, data2_sorted, cdf2, right=1)

    # 找到最大差异的索引和值
    max_diff_idx = np.argmax(np.abs(cdf1_interp - cdf2_interp))
    x_max_diff = all_values[max_diff_idx]
    y1_max_diff = cdf1_interp[max_diff_idx]
    y2_max_diff = cdf2_interp[max_diff_idx]

    # 在图上绘制最大差异线
    plt.vlines(x_max_diff, min(y1_max_diff, y2_max_diff), max(y1_max_diff, y2_max_diff),
               color='red', linestyle='--', label=f'最大分布差异 = {ks_statistic:.4f}')

    plt.title(f'K-S 检验: {name1} vs {name2}')
    plt.xlabel('累计释放百分比')
    plt.ylabel('累积概率')
    plt.text(0.5, 0.95, f'P值 = {p_value:.4f}', fontsize=sz-2, horizontalalignment='center', verticalalignment='top')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 1) # 释放比例范围是0-1
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"\n✅ K-S图已保存至: '{output_path}'")


if __name__ == '__main__':
    # ============================ 配置区域 ============================
    CV_RESULTS_PKL_PATH = "release_best_cv_results/cv_results_Hybrid_Autoregressive_Model_2_shot.pkl"
    OUTPUT_FOLDER = "release_feature_figures_zh"
    CV_RUN_INDEX = 0
    # ================================================================

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- 1. 加载并提取测试集数据 ---
    print("--- 步骤1: 加载并提取测试集预测与真实值 ---")
    try:
        cv_results = pd.read_pickle(CV_RESULTS_PKL_PATH)
        df_test = data_extraction(cv_results, CV_RUN_INDEX)
        print("数据加载成功。")
    except Exception as e:
        print(f"数据加载过程中出错: {e}"); exit()

    # 获取需要比较的两组数据
    predicted_values = df_test['Predicted_Release'].dropna().values
    experimental_values = df_test['Experimental_Release'].dropna().values

    # --- 2. 绘制K-S图 ---
    plot_ks_curve(
        predicted_values,
        experimental_values,
        name1='预测值',
        name2='实验值',
        output_path=os.path.join(OUTPUT_FOLDER, 'zh_ks_plot_prediction_vs_experiment.png')
    )