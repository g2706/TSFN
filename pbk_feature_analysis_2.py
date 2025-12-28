import os
import pickle
import pandas as pd
from pbk_ncv_try1 import NESTED_CV
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler


def generate_summary_plot(shap_values, X, filename='release_figures/cut6_or_mo_sup_1_SHAP_summary.png'):
    """生成、定制并保存SHAP摘要图。"""
    print("正在生成 SHAP 摘要图...")

    # 调用SHAP绘图函数
    shap.summary_plot(shap_values, X, feature_names=X.columns, show=False, plot_size=(10, 12))

    # 获取并定制图表对象
    fig, ax = plt.gcf(), plt.gca()
    ax.tick_params(labelsize=14)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=14)
    ax.set_title('Feature Importance - Fractional Drug Release', fontsize=16, weight="bold", pad=20)

    # 定制颜色条
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Feature value", fontsize=16)
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

    # 调用SHAP绘图函数
    shap.decision_plot(explainer.expected_value, shap_values[sample_slice],
                       feature_names=X.columns.tolist(), title=" ",
                       xlim=[0, 1], color_bar=False, show=False)

    # 获取并定制图表对象
    fig, ax = plt.gcf(), plt.gca()
    ax.tick_params(labelsize=16)
    ax.set_xlabel("Model Output: Fractional Drug Release", fontsize=16, weight="bold", color='black')

    # 保存并关闭图表
    plt.tight_layout()
    plt.savefig(filename, dpi=600, format='png', transparent=False)
    plt.close()
    print(f"决策图已保存至 {filename}")


if __name__ == '__main__':
    # 读取你转换后的 Excel 文件
    df = pd.read_excel("PBK_data/pbk_sup_1_data.xlsx")

    with open('PBK_Trained_models/pbk_sup_1_data_LGBM_model.pkl', 'rb') as f:
        LGBM_model = pickle.load(f)

    # LGBM_model=load_method.__self__.final_model
    print(dir(LGBM_model))

    # --- 全局设置 ---
    shap.initjs()
    os.makedirs("PBK_figures", exist_ok=True)

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
                          filename='PBK_figures/sup_1_SHAP_summary.png')

    # 2. 生成决策图 (为第2号样本)
    generate_decision_plot(explainer, shap_values, X,
                           sample_slice=slice(9, 21),
                           filename='PBK_figures/sup_1_Decision_plot_INDEX2_SLIDES.png')

    print("\n所有分析和绘图已完成！")