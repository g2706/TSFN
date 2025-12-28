import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Scikit-learn 和 Skorch
from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from skorch import NeuralNetRegressor
#没有遵循该原则：同一药物不能分别出现在测试集和训练集中

# --- 步骤1: 实现可复现性 ---
def set_seed(seed=42):
    """
    设置随机种子以确保结果可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为: {seed}")

# --- 步骤2: 定义GRU模型 (PyTorch部分) ---
class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(GRUPredictor, self).__init__()
        # Skorch 会自动处理输出维度，所以这里不需要 output_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1) # 输出固定为1

    def forward(self, x):
        # Skorch 会处理数据类型，但我们最好确保是 float
        x = x.to(torch.float32)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# --- 步骤3: 改造您的 NESTED_CV 类 ---
class NESTED_CV:
    def __init__(self, datafile="release_data/sup_4_release_data.xlsx", model_type='GRU'):
        # 构造函数现在只处理GRU模型
        self.df = pd.read_excel(datafile)
        self.SEQ_LENGTH = 3 # 序列长度

        if model_type == 'GRU':
            # 定义 Skorch 包装器
            self.user_defined_model = NeuralNetRegressor(
                module=GRUPredictor,
                module__input_size=0,  # 稍后在 input_target 中动态设置
                criterion=torch.nn.L1Loss,
                optimizer=torch.optim.Adam,
                device='cuda',
                # 默认训练参数，这些参数可以被 p_grid 覆盖
                max_epochs=100,
                lr=0.01,
                batch_size=32,
                verbose=0,  # 关闭skorch的训练日志
                # train_split=None # 我们在外部处理CV，所以skorch内部不需要再划分
            )
            # 为 GRU 和 Skorch 定义超参数搜索空间
            self.p_grid = {
                'max_epochs': [100, 150, 200],
                'lr': [0.01, 0.005, 0.001],
                'batch_size': [16, 32, 64],
                'module__hidden_size': [32, 64, 128],
                'module__num_layers': [1, 2, 3],
            }
        else:
            raise ValueError("此版本只支持 'GRU' 模型类型")

        # 初始化 Scalers
        self.feature_scaler = MinMaxScaler()
        self.release_scaler = MinMaxScaler()

    def input_target(self):
        """
        重写此方法以创建序列数据，而不是扁平化的表格数据。
        """
        print("正在创建序列数据...")
        static_feature_columns = [
            'Polymer_MW', 'LA/GA', 'Polymer_size', 'Zeta_Potential', 'PDI', 'EE', 'Drug_MW', 'LogP', 'HBD',
            'HBA', 'HA', 'RB', 'PSA', 'NAR', 'FSP3'
        ]

        self.df['drug_name']=self.df['drug_name'].str.strip().str.lower()

        # 1. 拟合 Scalers (在整个数据集上fit，这在实际部署前是常见做法)
        self.feature_scaler.fit(self.df[static_feature_columns])
        self.release_scaler.fit(self.df[['release_percentage']])

        # 2. 创建序列
        X_list, y_list, groups_list, E_list, T_list = [], [], [], [], []

        grouped_by_sample = self.df.groupby('sample_id')

        for sample_id, group in grouped_by_sample:
            # 缩放特征和释放值
            features_scaled = self.feature_scaler.transform(group[static_feature_columns])
            releases_scaled = self.release_scaler.transform(group[['release_percentage']])
            times = group['time'].values

            if len(group) > self.SEQ_LENGTH:
                for j in range(len(group) - self.SEQ_LENGTH):
                    # 准备输入序列的释放值和时间
                    seq_in_release = releases_scaled[j : j + self.SEQ_LENGTH].flatten()

                    # 准备输入序列的静态特征 (取第一个即可，因为它们都一样)
                    sample_feat = features_scaled[0]

                    combined_seq = []
                    # 使用“前向”时间差
                    for k in range(self.SEQ_LENGTH):
                        time_delta = times[j+k+1] - times[j+k]
                        current_step_input = np.concatenate(([seq_in_release[k]], sample_feat, [time_delta]))
                        combined_seq.append(current_step_input)

                    X_list.append(np.array(combined_seq))
                    # 目标值是序列结束后的下一个点
                    y_list.append(releases_scaled[j + self.SEQ_LENGTH])
                    # 为每个生成的序列分配其所属的组 (这里我们用 sample_id 作为组)
                    groups_list.append(sample_id)
                    # *** 保留 Bug 修复：为每个序列保存其对应的 sample_id 和时间点 ***
                    E_list.append(sample_id)
                    T_list.append(times[j + self.SEQ_LENGTH])

        self.X = np.array(X_list, dtype=np.float32)
        self.Y = np.array(y_list, dtype=np.float32)
        self.G = np.array(groups_list)
        # self.E = self.df['sample_id']
        # self.T = self.df['time']
        self.E = np.array(E_list)
        self.T = np.array(T_list)

        # 动态设置模型的 input_size
        input_size = self.X.shape[2]
        self.user_defined_model.set_params(module__input_size=input_size)
        print(f"序列数据创建完成。X shape: {self.X.shape}, Y shape: {self.Y.shape}, Groups shape: {self.G.shape}")
        print(f"模型输入的特征维度 (input_size) 设置为: {input_size}")
        print(f"共找到 {len(np.unique(self.G))} 个样本组用于交叉验证。")

    def cross_validation(self, input_value):
        # 这个方法几乎不需要修改，因为Skorch包装器使GRU表现得像一个sklearn模型！
        if input_value is None:
            NUM_TRIALS = 10
        else:
            NUM_TRIALS = input_value

        self.itr_number, self.outer_results, self.inner_results, self.model_params = [], [], [], []
        self.G_test_list, self.y_test_list, self.pred_list = [], [], []
        self.E_test_list, self.T_test_list = [], []
        # (为简化，移除了E和T的存储，因为它们在序列化后意义不大)

        for i in range(NUM_TRIALS):
            print(f"\n开始第 {i + 1}/{NUM_TRIALS} 次外层交叉验证...")
            cv_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i)

            for train_index, test_index in cv_outer.split(self.X, self.Y, self.G):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.Y[train_index], self.Y[test_index]
                G_train, G_test = self.G[train_index], self.G[test_index]
                E_train, E_test = self.E[train_index], self.E[test_index]
                T_train, T_test = self.T[train_index], self.T[test_index]

                self.G_test_list.append(G_test)
                self.y_test_list.append(self.release_scaler.inverse_transform(y_test)) # 存原始尺度的y
                E_test = np.array(E_test)  # prevents index from being brought from dataframe
                self.E_test_list.append(E_test)
                T_test = np.array(T_test)  # prevents index from being brought from dataframe
                self.T_test_list.append(T_test)

                cv_inner = GroupKFold(n_splits=10)

                # RandomizedSearchCV 现在可以无缝处理 Skorch 包装的 GRU 模型
                search = RSCV(self.user_defined_model, self.p_grid, n_iter=100, verbose=0,
                              scoring='neg_mean_absolute_error', cv=cv_inner, n_jobs=1, refit=True,
                              random_state=i) # n_iter 设为10以加速，实际可增加

                result = search.fit(X_train, y_train, groups=G_train)

                best_model = result.best_estimator_
                best_score = abs(result.best_score_)
                self.inner_results.append(best_score)

                yhat_scaled = best_model.predict(X_test)
                yhat_original = self.release_scaler.inverse_transform(yhat_scaled) # 转回原始尺度
                self.pred_list.append(yhat_original)

                acc = mean_absolute_error(self.y_test_list[-1], yhat_original)

                self.itr_number.append(i + 1)
                self.outer_results.append(acc)
                self.model_params.append(result.best_params_)

                print('\n################################################################\n\nSTATUS REPORT:')
                print(f'迭代 {i + 1}/{NUM_TRIALS} 完成')
                print(f'测试集 MAE : {acc:.5f}, 最佳验证集 Score: {best_score:.5f}')
                print(f'\n最佳模型参数:\n{result.best_params_}')
                print("\n################################################################\n ")

    def results(self):
        # 这个方法完全复用您的逻辑
        list_of_tuples = list(
            zip(self.itr_number, self.inner_results, self.outer_results, self.model_params, self.G_test_list,
                self.E_test_list, self.T_test_list, self.y_test_list, self.pred_list))
        CV_dataset = pd.DataFrame(list_of_tuples,
                                  columns=['Iter', 'Valid Score', 'Test Score', 'Model Parms', 'DP_Groups',
                                           "Experimental Index", "Time", 'Experimental_Release', 'Predicted_Release'])
        CV_dataset['Score_difference'] = abs(CV_dataset['Valid Score'] - CV_dataset['Test Score'])
        CV_dataset.sort_values(by=['Score_difference', 'Test Score'], ascending=True, inplace=True)
        self.CV_dataset = CV_dataset.reset_index(drop=True)

    def best_model(self):
        # 这个方法也完全复用您的逻辑
        best_model_params = self.CV_dataset.iloc[0, 3]
        best_model = self.user_defined_model.set_params(**best_model_params)
        # 最终模型在所有序列上训练
        self.final_model = best_model.fit(self.X, self.Y)

def run_NESTED_CV(name, CV):
    # 复用您的运行和保存逻辑
    model_instance = NESTED_CV(model_type=name)
    model_instance.input_target()
    model_instance.cross_validation(CV)
    model_instance.results()
    model_instance.best_model()

    # 创建目录
    os.makedirs("release_NESTED_CV_RESULTS", exist_ok=True)
    os.makedirs("release_Trained_models", exist_ok=True)

    # 保存结果
    result_filename = f"release_NESTED_CV_RESULTS/re_GRU_sequence_model_results.pkl"
    model_instance.CV_dataset.to_pickle(result_filename)
    print(f"\nCV 结果已保存至: {result_filename}")

    # 保存模型
    model_filename = f"release_Trained_models/re_GRU_sequence_model_final.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model_instance.final_model, file)
    print(f"最终模型已保存至: {model_filename}")

    return model_instance.CV_dataset, model_instance.final_model

if __name__ == '__main__':
    # --- 在主程序入口设置种子！ ---
    set_seed(42)

    # 运行流程，CV=5 表示运行5轮外层循环
    result, model = run_NESTED_CV("GRU", CV=2)

    if result is not None and model is not None:
        print("\n嵌套交叉验证流程完成！")
        print("\nCV 结果摘要:")
        # 显示更多列
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(result.head(3))
        print("\n最终训练好的模型:")
        print(model)
    else:
        print("\n嵌套交叉验证流程未能成功完成。")