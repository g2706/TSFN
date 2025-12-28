import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
import pickle

# ==================== 主执行流程 ====================
if __name__ == '__main__':
    # ============================ 配置区域 ============================
    DATA_FILE_PATH = "release_data/sup_6m_release_data.xlsx"
    FEATURES = ['time', 'Polymer_MW', 'LA/GA', 'Polymer_size', 'Zeta_Potential', 'PDI', 'EE',
                'Drug_MW', 'LogP', 'HBD', 'HBA', 'HA', 'RB', 'PSA', 'NAR', 'FSP3',
                'LR_MW', 'LR_LogP', 'LR_HBD', 'LR_HBA', 'EMW']
    TARGET = 'release_percentage'
    GROUP_BY_COL = 'sample_id'

    # 输出的数据文件名
    OUTPUT_DATA_FILENAME = "release_feature_figures/feature_analysis_data.pkl"
    # ================================================================

    # --- 数据加载与准备 ---
    print(f"--- 正在从 {DATA_FILE_PATH} 加载数据 ---")
    df = pd.read_excel(DATA_FILE_PATH).dropna(subset=FEATURES + [TARGET, GROUP_BY_COL])
    X = df[FEATURES]
    y = df[TARGET]
    groups = df[GROUP_BY_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=FEATURES)
    print(f"数据准备完成: {len(df)} 个数据点, {len(X.columns)} 个特征。")

    # --- 步骤一：获取初始特征重要性 ---
    print("\n--- 步骤1: 训练模型并获取初始特征重要性 ---")
    model_full = lgb.LGBMRegressor(random_state=42, verbose=-1)
    model_full.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model_full.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print("最重要的特征Top 5:")
    print(importance_df.head())

    # --- 步骤二：迭代性能评估 (核心分析) ---
    print("\n--- 步骤2: 迭代计算加入每个特征后的模型性能 (此过程可能需要较长时间) ---")
    sorted_features_list = importance_df['feature'].tolist()
    performance_results = []

    for k in range(1, len(sorted_features_list) + 1):
        top_k_features = sorted_features_list[:k]
        X_subset = X[top_k_features]
        model_subset = lgb.LGBMRegressor(random_state=42, verbose=-1)
        cv_splitter = GroupKFold(n_splits=10)
        scores = cross_val_score(model_subset, X_subset, y, cv=cv_splitter, scoring='r2', groups=groups, n_jobs=-1)

        performance_results.append({
            'num_features': k,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        })
        print(f"  > 使用前 {k} 个特征 - 平均 R²: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    performance_df = pd.DataFrame(performance_results)

    # --- 步骤三：保存结果到文件 ---
    print(f"\n--- 步骤3: 保存分析结果到文件 ---")

    # 创建一个字典来包含两个关键的DataFrame
    analysis_results = {
        'importance_df': importance_df,
        'performance_df': performance_df
    }

    with open(OUTPUT_DATA_FILENAME, 'wb') as f:
        pickle.dump(analysis_results, f)

    print(f"\n✅ 分析完成！绘图所需数据已保存至: '{OUTPUT_DATA_FILENAME}'")