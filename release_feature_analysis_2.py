import os
import pickle
import pandas as pd
from matplotlib import rcParams

from release_ncv_try1 import NESTED_CV
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
# --- 全局绘图参数设置 ---
sz=18

rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = sz+3       # 设置标题字体大小
rcParams['axes.titleweight'] = 'bold' # 设置标题加粗
rcParams['axes.labelsize'] = sz       # 设置x和y轴标签字体大小
rcParams['axes.labelweight'] = 'bold' # 设置x和y轴标签加粗
rcParams['xtick.labelsize'] = sz-3
rcParams['ytick.labelsize'] = sz-3
rcParams['legend.fontsize'] = sz-2


def generate_summary_plot(shap_values, X, filename='release_figures/cut6_or_mo_sup_1_SHAP_summary.png'):
    """生成、定制并保存SHAP摘要图。"""
    print("正在生成 SHAP 摘要图...")
    # 调用SHAP绘图函数
    shap.summary_plot(shap_values, X, feature_names=X.columns, show=False, plot_size=(8,8),max_display=len(X.columns))

    # 获取并定制图表对象
    fig, ax = plt.gcf(), plt.gca()
    ax.tick_params(labelsize=sz-3)
    ax.set_xlabel("SHAP Value", fontsize=sz,weight='bold')
    ax.set_title('Feature Importance - Fractional Drug Release', pad=20)

    # 定制颜色条
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=sz)
    cb_ax.set_ylabel("Feature value", fontsize=sz,weight='bold')
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)

    # 保存并关闭图表
    plt.tight_layout()
    plt.savefig(filename, dpi=600, format='png', transparent=False)
    plt.close()
    print(f"摘要图已保存至 {filename}")


def generate_decision_plot(explainer, shap_values, X, sample_slice, filename='release_figures/sup_1_Decision_plot.png'):
    """生成、定制并保存SHAP决策图。"""
    print("正在生成 SHAP 决策图...")
    plt.figure(figsize=(8,8))

# 调用SHAP绘图函数
    shap.decision_plot(explainer.expected_value, shap_values[sample_slice],
                       feature_names=X.columns.tolist(),feature_display_range=slice(None, None, -1), title=" ",
                       xlim=[0, 1], color_bar=False, show=False)

    # 获取并定制图表对象
    ax = plt.gca()
    ax.tick_params(labelsize=sz-3)
    ax.set_xlabel("Fractional Release", fontsize=sz, weight="bold", color='black')
    ax.set_title("Feature Decision Tree for Sample 2",fontsize=sz+3, weight="bold")
    # 保存并关闭图表
    plt.tight_layout()
    plt.savefig(filename, dpi=600, format='png', transparent=False)
    plt.close()
    print(f"决策图已保存至 {filename}")

def generate_force_plot(explainer, shap_values, X, sample_index, filename):
    """
    为单个样本生成、定制并保存SHAP力图。

    参数:
    explainer: 已经训练好的 SHAP 解释器。
    shap_values: 所有样本的 SHAP 值矩阵。
    X: 原始特征数据框 (用于获取特征名)。
    sample_index: 要解释的样本的索引 (一个整数)。
    filename: 保存图片的文件路径。
    """
    print(f"正在为样本索引 {sample_index} 生成 SHAP 力图...")

    # 调用SHAP绘图函数
    # 注意matplotlib=True使其能够被savefig保存
    shap.force_plot(explainer.expected_value, shap_values[sample_index,:],
                    feature_names=X.columns.tolist(),
                    matplotlib=True, show=False, figsize=(12, 4),
                    contribution_threshold=0.05)

    # 保存并关闭图表
    # tight_layout() 可以在此调整布局，但力图通常布局良好
    plt.savefig(filename, dpi=600, format='png', transparent=False, bbox_inches='tight')
    plt.close()
    print(f"力图已保存至 {filename}")



if __name__ == '__main__':
    # 读取你转换后的 Excel 文件
    df = pd.read_excel("release_data/sup_6m_release_data.xlsx")

    with open('release_Trained_models/sup_6_release_data_LGBM_model.pkl', 'rb') as f:
        LGBM_model = pickle.load(f)

    # LGBM_model=load_method.__self__.final_model
    print(dir(LGBM_model))

    # --- 全局设置 ---
    shap.initjs()
    os.makedirs("release_figures", exist_ok=True)

    # 定义特征矩阵 X (请确保这里的列与模型训练时完全一致)
    # X = df.drop(columns=['sample_id', 'drug_name', 'source', 'release_percentage',
    #                      'EMW', 'LR_HBA', 'LR_HBD', 'LR_LogP', 'LR_MW'])
    # X = df.drop(columns=['sample_id', 'drug_name', 'source', 'release_percentage',
    #                      'EMW', 'LR_HBA', 'LR_HBD', 'LR_LogP', 'LR_MW', 'LA/GA'])
    X = df.drop(columns=['sample_id', 'drug_name', 'source', 'release_percentage'])

    # 数据缩放
    stdScale = StandardScaler().fit(X)
    X_scale = stdScale.transform(X)

    # 初始化SHAP解释器并计算SHAP值
    explainer = shap.TreeExplainer(LGBM_model)
    shap_values = explainer.shap_values(X_scale)

    # 1. 生成摘要图
    generate_summary_plot(shap_values, X,
                          filename='release_feature_figures_ultra/sup_6m_SHAP_summary.png')

    # 2. 生成决策图 (为第2号样本)
    generate_decision_plot(explainer, shap_values, X,
                           sample_slice=slice(7, 15),
                           filename='release_feature_figures_ultra/sup_6m_Decision_plot_INDEX2_SLIDES.png')

    sam_index=300
    # 3. 生成力图
    generate_force_plot(explainer, shap_values, X,
                        sample_index=sam_index,
                        filename=f'release_feature_figures_ultra/sup_6m_force_plot_sample_{sam_index}.png')

    print("\n所有分析和绘图已完成！")