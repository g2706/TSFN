import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import r2_score
from sklearn.svm import SVR

# --- 全局绘图参数设置 ---
# (这部分保持不变)
sz = 18
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = sz + 3
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelsize'] = sz
rcParams['axes.labelweight'] = 'bold'
rcParams['xtick.labelsize'] = sz - 3
rcParams['ytick.labelsize'] = sz - 3
rcParams['legend.fontsize'] = sz - 2

# 注意：data_extraction 函数已不再需要，因为它用于解析.pkl文件，我们现在直接读取xlsx。

if __name__ == '__main__':
    # ============================ 配置区域 (请在此处修改) ============================

    # --- 体外 (In Vitro) 数据配置 ---
    # 请填写包含体外实验数据的Excel文件路径
    INVITRO_XLSX_PATH = "release_data/time_not_zero.xlsx"  # <--- 修改这里
    # 请填写该Excel文件中代表“体外释放值”的列名
    INVITRO_RELEASE_COL = "release_percentage"  # <--- 修改这里

    # --- 体内 (In Vivo) 数据配置 ---
    # 请填写包含体内实验数据的Excel文件路径
    INVIVO_XLSX_PATH = "release_data/sup_real_vivo_release_data.xlsx"  # <--- 修改这里 (如果需要)
    # 请填写该Excel文件中代表“体内释放值”的列名
    INVIVO_RELEASE_COL = "invivo_release"  # <--- 修改这里 (如果需要)

    # --- 输出配置 ---
    OUTPUT_FOLDER = "release_Correlation_Analysis"
    OUTPUT_FIGURE_FILENAME = "correlation_experimental_invitro_vs_invivo.png"
    # =================================================================================

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- 步骤 1: 数据加载与准备 ---
    print("--- 步骤1: 加载体外和体内实验数据 ---")
    try:
        # 加载体外实验数据
        df_invitro = pd.read_excel(INVITRO_XLSX_PATH)
        required_invitro_cols = ['sample_id', 'time', INVITRO_RELEASE_COL]
        if not all(col in df_invitro.columns for col in required_invitro_cols):
            raise ValueError(
                f"错误: 体外(in vitro)数据文件 '{INVITRO_XLSX_PATH}' 缺少必要的列: {required_invitro_cols}")

        # 加载体内实验数据
        df_invivo = pd.read_excel(INVIVO_XLSX_PATH)
        required_invivo_cols = ['sample_id', 'time', INVIVO_RELEASE_COL]
        if not all(col in df_invivo.columns for col in required_invivo_cols):
            raise ValueError(f"错误: 体内(in vivo)数据文件 '{INVIVO_XLSX_PATH}' 缺少必要的列: {required_invivo_cols}")

        print("数据加载成功。")
    except Exception as e:
        print(f"数据准备过程中出错: {e}");
        exit()

    # --- 步骤 2: 通过插值对齐数据 (反向线性插值：估算体内值) ---
    print("\n--- 步骤2: 通过插值对齐数据 (反向线性插值：估算体内值) ---")
    aligned_data = []
    # 找到体外和体内数据集中共有的 'sample_id'
    common_sample_ids = set(df_invitro['sample_id']) & set(df_invivo['sample_id'])

    if not common_sample_ids:
        print("错误：体外和体内数据文件中没有共同的 'sample_id'，无法进行对齐。")
        exit()

    for sid in common_sample_ids:
        invitro_curve = df_invitro[df_invitro['sample_id'] == sid].sort_values('time')
        invivo_curve = df_invivo[df_invivo['sample_id'] == sid].sort_values('time')

        # 线性插值至少需要2个已知点来定义一条线段
        if len(invivo_curve) < 2 or invitro_curve.empty:
            print(f"警告: 样本ID {sid} 的体内数据点少于2个，无法进行线性插值，将跳过此样本。")
            continue

        # 使用 np.interp 进行线性插值，估算在“体外时间点”上的“体内释放值”
        interpolated_invivo_release = np.interp(
            invitro_curve['time'],  # 参数1 (x): 我们想要估算值的那些新X坐标 (即，体外时间点)
            invivo_curve['time'],  # 参数2 (xp): 我们已知的原始X坐标 (即，体内时间点)
            invivo_curve[INVIVO_RELEASE_COL]  # 参数3 (fp): 与原始X坐标对应的原始Y坐标 (即，体内释放值)
        )

        # 将对齐后的数据配对并存储
        for i in range(len(invitro_curve)):
            aligned_data.append({
                'sample_id': sid,
                'time': invitro_curve['time'].iloc[i],
                'invitro_release': invitro_curve[INVITRO_RELEASE_COL].iloc[i],  # 原始的体外实验值
                'invivo_release': interpolated_invivo_release[i]  # 通过线性插值估算出的体内值
            })

    df_aligned = pd.DataFrame(aligned_data)
    if df_aligned.empty: print("\n错误: 未能生成任何对齐的数据点。请检查 'sample_id' 和 'time' 数据。"); exit()
    print(f"成功生成了 {len(df_aligned)} 个对齐的数据点。")

    # --- 步骤 3: 使用多种模型进行拟合并评估R² ---
    print("\n--- 步骤3: 使用多种模型进行拟合与评估 ---")
    # x_data 现在是体外实验值, y_data 是体内实验值
    x_data = df_aligned['invitro_release'].values.reshape(-1, 1)
    y_data = df_aligned['invivo_release'].values

    # 1. 线性模型
    m, b = np.polyfit(x_data.flatten(), y_data, 1)
    y_pred_linear = m * x_data.flatten() + b
    r2_linear = r2_score(y_data, y_pred_linear)
    print(f"  - 线性模型 R²: {r2_linear:.4f} (y = {m:.2f}x + {b:.2f})")

    # 2. 多项式回归 (2阶)
    poly_coeffs = np.polyfit(x_data.flatten(), y_data, 2)
    poly_model = np.poly1d(poly_coeffs)
    y_pred_poly = poly_model(x_data.flatten())
    r2_poly = r2_score(y_data, y_pred_poly)
    print(f"  - 二阶多项式模型 R²: {r2_poly:.4f}")

    # 3. 支持向量回归 (SVR)
    svr = SVR(kernel='rbf')
    svr.fit(x_data, y_data)
    y_pred_svr = svr.predict(x_data)
    r2_svr = r2_score(y_data, y_pred_svr)
    print(f"  - SVR (RBF核) 模型 R²: {r2_svr:.4f}")

    # --- 步骤 4: 绘制并保存对比图 ---
    print("\n--- 步骤4: 生成拟合曲线对比图 ---")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_data, y_data, alpha=0.8, edgecolors='k', label='(In Vitro, In Vivo) Release Point')

    # 为了绘图，我们需要对x轴排序
    sort_axis = np.argsort(x_data.flatten())

    # 绘制拟合曲线
    ax.plot(x_data[sort_axis], y_pred_linear[sort_axis], 'r-', linewidth=2, label=f'Linear Fitting (R²={r2_linear:.3f})')
    ax.plot(x_data[sort_axis], y_pred_poly[sort_axis], 'g--', linewidth=2, label=f'Second-Order Polynomial Fitting (R²={r2_poly:.3f})')
    ax.plot(x_data[sort_axis], y_pred_svr[sort_axis], 'm-.', linewidth=2, label=f'SVR Fitting (R²={r2_svr:.3f})')

    ax.set_xlabel('In Vitro Experimental Release')
    ax.set_ylabel('In Vivo Experimental Release')
    ax.set_title('In Vitro-In Vivo Correlation (Exp vs. Exp)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FIGURE_FILENAME)
    plt.savefig(output_path, dpi=600)
    plt.close(fig)

    print(f"\n✅ 分析完成！包含多种拟合曲线的图表已保存至: '{output_path}'")