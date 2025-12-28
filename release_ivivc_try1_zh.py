import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
# <--- 新增: 导入SVR模型
from sklearn.svm import SVR

# ==================== 字体设置代码 ====================
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
# ==========================================================
# --- 全局绘图参数设置 ---
sz=18

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
    CV_RESULTS_PKL_PATH = "release_best_cv_results/cv_results_Hybrid_Autoregressive_Model_2_shot.pkl"
    EXTERNAL_XLSX_PATH = "release_data/sup_real_vivo_release_data.xlsx" # 请替换为您的文件路径
    EXTERNAL_DATA_COLUMN_NAME = "invivo_release"
    OUTPUT_FOLDER = "release_feature_figures_zh"
    OUTPUT_FIGURE_FILENAME = "zh_correlation_nonlinear_fits_2_shot.png"
    # ================================================================

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- 步骤 1 & 2: 数据加载与准备 ---
    print("--- 步骤1-2: 加载与准备数据 ---")
    try:
        cv_results = pd.read_pickle(CV_RESULTS_PKL_PATH)
        df_invitro_test = data_extraction(cv_results, 0)
        df_invitro_test.rename(columns={'Experimental Index': 'sample_id', 'Time': 'time'}, inplace=True)
        df_external = pd.read_excel(EXTERNAL_XLSX_PATH)
        required_cols = ['sample_id', 'time', EXTERNAL_DATA_COLUMN_NAME]
        if not all(col in df_external.columns for col in required_cols):
            raise ValueError(f"错误: xlsx文件缺少必要的列: {required_cols}")
        print("数据加载成功。")
    except Exception as e:
        print(f"数据准备过程中出错: {e}"); exit()

    # --- 步骤 3: 通过插值对齐数据 ---
    print("\n--- 步骤3: 通过插值对齐数据 ---")
    aligned_data = []
    common_sample_ids = set(df_invitro_test['sample_id']) & set(df_external['sample_id'])
    for sid in common_sample_ids:
        invitro_curve = df_invitro_test[df_invitro_test['sample_id'] == sid].sort_values('time')
        external_curve = df_external[df_external['sample_id'] == sid].sort_values('time')
        if invitro_curve.empty or external_curve.empty: continue
        interpolated_invitro_release = np.interp(external_curve['time'], invitro_curve['time'], invitro_curve['Predicted_Release'])
        for i in range(len(external_curve)):
            aligned_data.append({'sample_id': sid, 'time': external_curve['time'].iloc[i],
                                 'external_release': external_curve[EXTERNAL_DATA_COLUMN_NAME].iloc[i],
                                 'predicted_invitro_release': interpolated_invitro_release[i]})
    df_aligned = pd.DataFrame(aligned_data)
    if df_aligned.empty: print("\n错误: 未能生成任何对齐的数据点。"); exit()
    print(f"成功生成了 {len(df_aligned)} 个对齐的数据点。")

    # --- 步骤 4: 使用多种模型进行拟合并评估R² ---
    print("\n--- 步骤4: 使用多种模型进行拟合与评估 ---")
    x_data = df_aligned['predicted_invitro_release'].values.reshape(-1, 1)
    y_data = df_aligned['external_release'].values

    # 1. 线性模型
    m, b = np.polyfit(x_data.flatten(), y_data, 1)
    y_pred_linear = m * x_data.flatten() + b
    r2_linear = r2_score(y_data, y_pred_linear)
    print(f"  - Linear model R²: {r2_linear:.4f} (y = {m:.2f}x + {b:.2f})")

    # 2. 多项式回归 (2阶)
    poly_coeffs = np.polyfit(x_data.flatten(), y_data, 2)
    poly_model = np.poly1d(poly_coeffs)
    y_pred_poly = poly_model(x_data.flatten())
    r2_poly = r2_score(y_data, y_pred_poly)
    print(f"  - Second-order polynomial model R²: {r2_poly:.4f}")

    # 3. 支持向量回归 (SVR)
    svr = SVR(kernel='rbf') # RBF核可以拟合复杂的非线性关系
    svr.fit(x_data, y_data)
    y_pred_svr = svr.predict(x_data)
    r2_svr = r2_score(y_data, y_pred_svr)
    print(f"  - SVR (RBF核) 模型 R²: {r2_svr:.4f}")


    # --- 步骤 5: 绘制并保存对比图 ---
    print("\n--- 步骤5: 生成拟合曲线对比图 ---")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_data, y_data, alpha=0.8, edgecolors='k', label='释放点')

    # 为了绘图，我们需要对x轴排序
    sort_axis = np.argsort(x_data.flatten())

    # 绘制拟合曲线
    ax.plot(x_data[sort_axis], y_pred_linear[sort_axis], 'r-', linewidth=2, label=f'线性拟合 (R²={r2_linear:.3f})')
    ax.plot(x_data[sort_axis], y_pred_poly[sort_axis], 'g--', linewidth=2, label=f'二阶多项式拟合 (R²={r2_poly:.3f})')
    ax.plot(x_data[sort_axis], y_pred_svr[sort_axis], 'm-.', linewidth=2, label=f'支持向量机回归拟合 (R²={r2_svr:.3f})')

    ax.set_xlabel('体外预测释放值')
    ax.set_ylabel('体内实验释放值')
    ax.set_title('体内体外相关性 (预测值 vs. 实验值)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FIGURE_FILENAME)
    plt.savefig(output_path, dpi=600)
    plt.close(fig)

    print(f"\n✅ 分析完成！包含多种拟合曲线的图表已保存至: '{output_path}'")