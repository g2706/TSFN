import pandas as pd

# 假设你的文件名是这个，请替换成你实际的文件名
# filename = "release_NESTED_CV_RESULTS/cut5_sup_6m_release_data_XGB.pkl"
# filename = "release_NESTED_CV_RESULTS_GRU/or_GRU_model_results.pkl"
filename = "release_best_cv_results/cv_results_Hybrid_Autoregressive_Model_1_shot.pkl"
try:

    # 1. 使用 pandas 的 read_pickle 来读取这个文件
    cv_results_df = pd.read_pickle(filename)

    # 2. 现在 cv_results_df 就是一个DataFrame，你可以像操作其他表格数据一样操作它
    print("--- 成功从PKL文件中读取DataFrame！ ---")

    # 打印DataFrame的形状 (行, 列)
    print(f"\nDataFrame的形状: {cv_results_df.shape}")

    # 打印前5行，来查看数据结构和内容
    print("\nDataFrame的前5行内容:")
    print(cv_results_df.head())

    # 打印所有的列名，了解它记录了哪些信息
    print("\nDataFrame包含的列名:")
    print(cv_results_df.columns.tolist())

    # 每一轮的测试分数
    print("\n每一轮的测试分数:")
    for i, score in enumerate(cv_results_df['Test Score']):
        print(f"第{i+1}轮测试分数: {score:.4f}")

    # 计算测试分数的平均值和标准差
    mean_test_score = cv_results_df['Test Score'].mean()
    std_test_score = cv_results_df['Test Score'].std()

    print(f"\n模型在10轮10折交叉验证中的平均测试分数: {mean_test_score:.4f}")
    print(f"模型测试分数的标准差: {std_test_score:.4f}")

except FileNotFoundError:
    print(f"错误：找不到文件 '{filename}'。请确保文件名和路径正确。")
except Exception as e:
    print(f"读取文件时发生错误: {e}")