import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from skorch import NeuralNetRegressor
#已遵循该原则：同一药物不能分别出现在测试集和训练集中

# --- 实现可复现性 ---
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为: {seed}")

# --- 定义GRU模型 ---
class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(GRUPredictor, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.to(torch.float32)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# --- 定义LSTM模型 ---
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.rnn = nn.LSTM( # 这里使用 nn.LSTM
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.to(torch.float32)
        # LSTM 返回 (out, (h_n, c_n))，我们只需要out
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# --- NESTED_CV 用于交叉验证 ---
class NESTED_CV:
    def __init__(self, datafile="release_data/sup_4_release_data.xlsx", model_type='GRU'):
        self.df = pd.read_excel(datafile)
        self.SEQ_LENGTH = 3
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n模型将运行在 '{device}' 设备上。")

        if model_type == 'GRU':
            print("已选择 GRU 模型。")
            model_module = GRUPredictor
            self.p_grid = {
                'max_epochs': [100, 150, 200], 'lr': [0.01, 0.005, 0.001],
                'batch_size': [16, 32, 64], 'module__hidden_size': [32, 64, 128],
                'module__num_layers': [1, 2, 3],
            }
        elif model_type == 'LSTM':
            print("已选择 LSTM 模型。")
            model_module = LSTMPredictor
            # LSTM 和 GRU 的超参数空间通常可以共用
            self.p_grid = {
                'max_epochs': [100, 150, 200], 'lr': [0.01, 0.005, 0.001],
                'batch_size': [16, 32, 64], 'module__hidden_size': [32, 64, 128],
                'module__num_layers': [1, 2, 3],
            }

        else:
            raise ValueError(f"不支持的模型类型: '{model_type}'. 请选择 'GRU' 或 'LSTM'.")

        self.user_defined_model = NeuralNetRegressor(
            module=model_module,
            module__input_size=0,
            criterion=torch.nn.L1Loss,
            optimizer=torch.optim.Adam,
            device=device,
            max_epochs=100,
            lr=0.01,
            batch_size=32,
            verbose=0,
        )

        self.feature_scaler = MinMaxScaler()
        self.release_scaler = MinMaxScaler()

    def input_target(self):
        print("正在创建序列数据...")
        static_feature_columns = [
            'Polymer_MW', 'LA/GA', 'Polymer_size', 'Zeta_Potential', 'PDI', 'EE', 'Drug_MW', 'LogP', 'HBD',
            'HBA', 'HA', 'RB', 'PSA', 'NAR', 'FSP3'
        ]

        self.df['drug_name']=self.df['drug_name'].str.strip().str.lower()
        self.feature_scaler.fit(self.df[static_feature_columns])
        self.release_scaler.fit(self.df[['release_percentage']])

        X_list, y_list, groups_list, E_list, T_list = [], [], [], [], []

        # *** 核心修改 ***
        # 仍然按 sample_id 遍历来确保样本不被拆散
        # 在此基础上使用 drug_name 作为分组的标签
        grouped_by_sample = self.df.groupby('sample_id')

        for sample_id, group in grouped_by_sample:
            features_scaled = self.feature_scaler.transform(group[static_feature_columns])
            releases_scaled = self.release_scaler.transform(group[['release_percentage']])
            times = group['time'].values

            # 获取当前样本的 drug_name (一个样本只有一个drug_name)
            drug_name = group['drug_name'].iloc[0]

            if len(group) > self.SEQ_LENGTH:
                for j in range(len(group) - self.SEQ_LENGTH):
                    seq_in_release = releases_scaled[j : j + self.SEQ_LENGTH].flatten()
                    sample_feat = features_scaled[0]

                    combined_seq = []
                    for k in range(self.SEQ_LENGTH):
                        time_delta = times[j+k+1] - times[j+k]
                        current_step_input = np.concatenate(([seq_in_release[k]], sample_feat, [time_delta]))
                        combined_seq.append(current_step_input)

                    X_list.append(np.array(combined_seq))
                    y_list.append(releases_scaled[j + self.SEQ_LENGTH])

                    # *** 将 drug_name 作为组标签加入列表 ***
                    groups_list.append(drug_name)

                    # *** 保留 Bug 修复：为每个序列保存其对应的 sample_id 和时间点 ***
                    E_list.append(sample_id)
                    T_list.append(times[j + self.SEQ_LENGTH])

        self.X = np.array(X_list, dtype=np.float32)
        self.Y = np.array(y_list, dtype=np.float32)

        # GroupKFold/GroupShuffleSplit 需要数字作为group标签，我们将drug_name字符串转换为数字ID
        # pandas.factorize 会返回一个数字编码数组和一个唯一值索引
        self.G, self.group_labels = pd.factorize(np.array(groups_list))
        self.E = np.array(E_list)
        self.T = np.array(T_list)

        input_size = self.X.shape[2]
        self.user_defined_model.set_params(module__input_size=input_size)
        print(f"序列数据创建完成。X shape: {self.X.shape}, Y shape: {self.Y.shape}, Groups shape: {self.G.shape}")
        print(f"模型输入的特征维度 (input_size) 设置为: {input_size}")
        print(f"共找到 {len(np.unique(self.G))} 个药物组用于交叉验证。")

    # 它们会自动使用新的 self.G (基于drug_name)来进行分组
    def cross_validation(self, input_value):
        if input_value is None: NUM_TRIALS = 10
        else: NUM_TRIALS = input_value

        self.itr_number, self.outer_results, self.inner_results, self.model_params = [], [], [], []
        self.G_test_list, self.y_test_list, self.pred_list = [], [], []
        self.E_test_list, self.T_test_list = [], []

        for i in range(NUM_TRIALS):
            print(f"\n开始第 {i + 1}/{NUM_TRIALS} 次外层交叉验证 (按 drug_name 分组)...")
            cv_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i)

            for train_index, test_index in cv_outer.split(self.X, self.Y, self.G):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.Y[train_index], self.Y[test_index]
                G_train, G_test = self.G[train_index], self.G[test_index]
                E_train, E_test = self.E[train_index], self.E[test_index]
                T_train, T_test = self.T[train_index], self.T[test_index]

                self.G_test_list.append(self.group_labels[G_test]) # 存原始的drug_name
                self.y_test_list.append(self.release_scaler.inverse_transform(y_test))
                E_test = np.array(E_test)  # prevents index from being brought from dataframe
                self.E_test_list.append(E_test)
                T_test = np.array(T_test)  # prevents index from being brought from dataframe
                self.T_test_list.append(T_test)

                cv_inner = GroupKFold(n_splits=10)

                search = RSCV(self.user_defined_model, self.p_grid, n_iter=100, verbose=0,
                              scoring='neg_mean_absolute_error', cv=cv_inner, n_jobs=1, refit=True,
                              random_state=i)

                result = search.fit(X_train, y_train, groups=G_train)

                best_model = result.best_estimator_
                best_score = abs(result.best_score_)
                self.inner_results.append(best_score)

                yhat_scaled = best_model.predict(X_test)
                yhat_original = self.release_scaler.inverse_transform(yhat_scaled)
                self.pred_list.append(yhat_original)

                acc = mean_absolute_error(self.y_test_list[-1], yhat_original)

                self.itr_number.append(i + 1)
                self.outer_results.append(acc)
                self.model_params.append(result.best_params_)

                print('\n################################################################\n\nSTATUS REPORT:')
                print(f'迭代 {i + 1}/{NUM_TRIALS} 完成')
                print(f'测试集 MAE (原始尺度): {acc:.5f}, 最佳验证集 Score : {best_score:.5f}')
                print(f'\n最佳模型参数:\n{result.best_params_}')
                print("\n################################################################\n ")

    def results(self):
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
        best_model_params = self.CV_dataset.iloc[0, 3]
        best_model = self.user_defined_model.set_params(**best_model_params)
        self.final_model = best_model.fit(self.X, self.Y)


def run_NESTED_CV(name, CV):
    model_instance = NESTED_CV(model_type=name)
    model_instance.input_target()
    model_instance.cross_validation(CV)
    model_instance.results()
    model_instance.best_model()

    os.makedirs(f"release_NESTED_CV_RESULTS_{name}", exist_ok=True)
    os.makedirs(f"release_Trained_models_{name}", exist_ok=True)

    result_filename = f"release_NESTED_CV_RESULTS_{name}/cut5_{name}_sequence_model_results.pkl"
    model_instance.CV_dataset.to_pickle(result_filename)
    print(f"\nCV 结果已保存至: {result_filename}")

    model_filename = f"release_Trained_models_{name}/cut5_{name}_sequence_model_final.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model_instance.final_model, file)
    print(f"最终模型已保存至: {model_filename}")

    return model_instance.CV_dataset, model_instance.final_model


if __name__ == '__main__':
    set_seed(42)
    MODEL_TO_RUN="LSTM"
    if torch.cuda.is_available(): print(f"GPU is available! Device: {torch.cuda.get_device_name(0)}")
    else: print("GPU not available. Code will run on CPU.")

    result, model = run_NESTED_CV(MODEL_TO_RUN, CV=10)

    if result is not None and model is not None:
        print("\n嵌套交叉验证流程完成！")
        print("\nCV 结果摘要:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(result.head(3))
        print(f"\n最终训练好的{MODEL_TO_RUN}模型:")
        print(model)
    else:
        print("\n嵌套交叉验证流程未能成功完成。")