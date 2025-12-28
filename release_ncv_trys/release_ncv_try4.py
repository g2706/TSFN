import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import random
import os
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 使用了合适的时间差数据



# 确保安装了openpyxl: pip install openpyxl
# ==============================================================================
# 步骤 1: 数据加载 (无变化)
# ==============================================================================
def load_and_process_data(file_path):
    """从XLSX文件加载数据"""
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError: return None, None, None, None
    static_feature_columns = [
        'Polymer_MW', 'LA/GA', 'Polymer_size', 'Zeta_Potential',  'PDI', 'EE', 'Drug_MW', 'LogP', 'HBD',
        'HBA', 'HA', 'RB', 'PSA', 'NAR', 'FSP3'
    ]
    grouped = df.groupby('sample_id')
    all_features, all_curves, all_times, sample_ids = [], [], [], []
    for sample_id, group in grouped:
        all_features.append(group[static_feature_columns].iloc[0].values)
        all_times.append(group['time'].values)
        all_curves.append(group['release_percentage'].values)
        sample_ids.append(sample_id)
    print(f"数据加载完成！共找到 {len(all_features)} 个独立样本。")
    return np.array(all_features, dtype=np.float32), all_curves, all_times, sample_ids

# ==============================================================================
# 步骤 2: Dataset 定义 (核心修改)
# ==============================================================================
class DrugReleaseDataset(Dataset):
    def __init__(self, features, times, releases, sequence_length=3):
        self.X, self.y = [], []
        for i in range(len(features)):
            sample_feat, sample_time, sample_release = features[i], times[i], releases[i]

            # 确保样本长度足够创建至少一个序列
            # 因为我们需要 time[j+k+1]，所以总长度需要比 sequence_length + 1 还多
            if len(sample_release) > sequence_length:
                for j in range(len(sample_release) - sequence_length):
                    # --- 准备输入和输出序列 ---
                    seq_in_release = sample_release[j : j + sequence_length]
                    # 目标值 (y) 是输入序列结束后的下一个点
                    seq_out_release = sample_release[j + sequence_length]

                    combined_seq = []
                    # --- 核心修改：计算前向时间差 ---
                    # 遍历输入序列的每个点
                    for k in range(sequence_length):
                        # 计算当前点(j+k)到下一个点(j+k+1)的时间差
                        time_delta = sample_time[j+k+1] - sample_time[j+k]

                        # 组合特征: [当前释放值, 静态特征, 前向时间差]
                        current_step_input = np.concatenate(([seq_in_release[k]], sample_feat, [time_delta]))
                        combined_seq.append(current_step_input)

                    self.X.append(np.array(combined_seq))
                    self.y.append(seq_out_release)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).float()


# 其他辅助函数和模型定义 (无变化)
def pad_collate_fn(batch):
    (xx, yy) = zip(*batch)
    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy = torch.stack(yy, 0)
    return xx_pad, yy

class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUPredictor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x); return self.fc(out[:, -1, :])

# ==============================================================================
# 步骤 3: 预测函数 (核心修改)
# ==============================================================================
def predict_full_curve_irregular(model, initial_releases, initial_times, static_features, future_times, seq_len):
    model.eval()
    predicted_releases = list(initial_releases)
    current_release_seq = list(initial_releases)
    current_time_seq = list(initial_times)

    with torch.no_grad():
        for i in range(len(future_times)):
            # 准备本次预测的输入序列
            combined_input_seq = []

            # --- 核心修改：为输入序列构建前向时间差 ---
            for k in range(seq_len):
                # 如果不是序列的最后一个点，时间差是序列内下一个点和当前点之差
                if k < seq_len - 1:
                    time_delta = current_time_seq[k+1] - current_time_seq[k]
                # 如果是序列的最后一个点，时间差是我们要预测的点和当前点之差
                else:
                    time_delta = future_times[i] - current_time_seq[k]

                # 组合特征
                current_step_input = np.concatenate(([current_release_seq[k]], static_features, [time_delta]))
                combined_input_seq.append(current_step_input)

            # 转换为张量并预测
            input_tensor = torch.from_numpy(np.array([combined_input_seq])).float()
            next_release_pred = model(input_tensor).item()

            # 更新序列用于下一次预测
            predicted_releases.append(next_release_pred)
            current_release_seq.pop(0)
            current_time_seq.pop(0)
            current_release_seq.append(next_release_pred)
            current_time_seq.append(future_times[i]) # 将下一个要预测的时间点加入序列

    return np.array(predicted_releases)

# 训练和评估函数 (修复bug后的版本，无新变化)
def train_and_evaluate(model, train_loader, val_loader, params):
    """在一个训练集上训练，并在一个验证集上评估"""
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    model.train()
    for epoch in range(params['epochs']):
        for x_padded, y_true in train_loader:
            y_pred = model(x_padded)
            loss = criterion(y_pred, y_true.unsqueeze(1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x_padded, y_true in val_loader:
            y_pred = model(x_padded)
            loss = criterion(y_pred, y_true.unsqueeze(1))
            total_val_loss += loss.item() * x_padded.size(0)
    if len(val_loader.dataset) == 0: return float('inf')
    return total_val_loss / len(val_loader.dataset)


# ==============================================================================
# 主程序：嵌套交叉验证 (无变化)
# ==============================================================================
if __name__ == '__main__':
    # --- 全局参数 ---
    XLSX_FILE_PATH = 'release_data/sup_4_release_data.xlsx'
    SEQ_LENGTH = 3
    NUM_TRIALS = 10
    N_ITER_SEARCH = 100
    K_FOLDS_INNER = 10

    # --- 1. 加载数据 ---
    all_sample_features, all_release_curves, all_time_vectors, all_sample_ids = load_and_process_data(XLSX_FILE_PATH)

    if all_sample_features is not None:
        # --- 2. 定义超参数搜索空间 ---
        p_grid = {
            'hidden_size': [32, 64, 128], 'num_layers': [1, 2, 3],
            'lr': [0.01, 0.005, 0.001], 'batch_size': [16, 32],
            'epochs': [100]
        }

        outer_results, best_params_list = [], []

        # --- 3. 外层交叉验证循环 ---
        print(f"--- 开始嵌套交叉验证 (共 {NUM_TRIALS} 轮外层循环) ---\n")
        sample_indices = np.arange(len(all_sample_ids))

        for i in range(NUM_TRIALS):
            print(f"\n################## 外层循环 {i+1}/{NUM_TRIALS} ##################")
            outer_train_indices, outer_test_indices = train_test_split(sample_indices, test_size=0.2, random_state=i)

            # 数据缩放
            feature_scaler, release_scaler = MinMaxScaler(), MinMaxScaler()
            feat_outer_train = all_sample_features[outer_train_indices]
            curves_outer_train = [all_release_curves[j] for j in outer_train_indices]
            scaled_feat_outer_train = feature_scaler.fit_transform(feat_outer_train)
            release_scaler.fit(np.concatenate(curves_outer_train).reshape(-1, 1))

            # --- 内层循环：超参数搜索 ---
            print(f"--- 开始内层 {K_FOLDS_INNER}-折交叉验证以搜索超参数 ({N_ITER_SEARCH} 次尝试) ---")
            hyperparam_search_results = []

            for j in range(N_ITER_SEARCH):
                params = {k: random.choice(v) for k, v in p_grid.items()}
                kf_inner = KFold(n_splits=K_FOLDS_INNER, shuffle=True, random_state=j)
                inner_fold_losses = []

                for inner_train_idx_of_idx, inner_val_idx_of_idx in kf_inner.split(outer_train_indices):
                    # 省略数据准备细节，逻辑同之前
                    inner_train_indices = outer_train_indices[inner_train_idx_of_idx]
                    inner_val_indices = outer_train_indices[inner_val_idx_of_idx]

                    feat_inner_train, curves_inner_train_raw, times_inner_train = all_sample_features[inner_train_indices], [all_release_curves[k] for k in inner_train_indices], [all_time_vectors[k] for k in inner_train_indices]
                    feat_inner_val, curves_inner_val_raw, times_inner_val = all_sample_features[inner_val_indices], [all_release_curves[k] for k in inner_val_indices], [all_time_vectors[k] for k in inner_val_indices]

                    scaled_feat_inner_train, scaled_curves_inner_train = feature_scaler.transform(feat_inner_train), [release_scaler.transform(c.reshape(-1,1)).flatten() for c in curves_inner_train_raw]
                    scaled_feat_inner_val, scaled_curves_inner_val = feature_scaler.transform(feat_inner_val), [release_scaler.transform(c.reshape(-1,1)).flatten() for c in curves_inner_val_raw]

                    train_ds_inner, val_ds_inner = DrugReleaseDataset(scaled_feat_inner_train, times_inner_train, scaled_curves_inner_train, SEQ_LENGTH), DrugReleaseDataset(scaled_feat_inner_val, times_inner_val, scaled_curves_inner_val, SEQ_LENGTH)

                    if not train_ds_inner or not val_ds_inner: continue

                    train_loader_inner, val_loader_inner = DataLoader(train_ds_inner, batch_size=params['batch_size'], shuffle=True, collate_fn=pad_collate_fn), DataLoader(val_ds_inner, batch_size=params['batch_size'], shuffle=False, collate_fn=pad_collate_fn)

                    INPUT_SIZE = 1 + all_sample_features.shape[1] + 1
                    model = GRUPredictor(INPUT_SIZE, params['hidden_size'], params['num_layers'], 1)

                    fold_loss = train_and_evaluate(model, train_loader_inner, val_loader_inner, params)
                    inner_fold_losses.append(fold_loss)

                avg_val_loss = np.mean(inner_fold_losses)
                hyperparam_search_results.append({'params': params, 'loss': avg_val_loss})
                print(f"  > 尝试 {j+1}/{N_ITER_SEARCH}: params={params}, avg_val_loss={avg_val_loss:.6f}")

            # 选出最佳超参数并评估
            best_inner_result = min(hyperparam_search_results, key=lambda x: x['loss'])
            best_params = best_inner_result['params']
            best_params_list.append(best_params)
            print(f"\n--- 外层循环 {i+1} 的最佳超参数: {best_params} (验证MSE: {best_inner_result['loss']:.6f}) ---")

            # 准备最终训练和测试数据
            times_outer_train = [all_time_vectors[j] for j in outer_train_indices]
            scaled_curves_outer_train = [release_scaler.transform(c.reshape(-1,1)).flatten() for c in curves_outer_train]
            feat_outer_test, curves_outer_test_raw, times_outer_test = all_sample_features[outer_test_indices], [all_release_curves[j] for j in outer_test_indices], [all_time_vectors[j] for j in outer_test_indices]
            scaled_feat_outer_test, scaled_curves_outer_test = feature_scaler.transform(feat_outer_test), [release_scaler.transform(c.reshape(-1,1)).flatten() for c in curves_outer_test_raw]

            final_train_ds, final_test_ds = DrugReleaseDataset(scaled_feat_outer_train, times_outer_train, scaled_curves_outer_train, SEQ_LENGTH), DrugReleaseDataset(scaled_feat_outer_test, times_outer_test, scaled_curves_outer_test, SEQ_LENGTH)
            final_train_loader, final_test_loader = DataLoader(final_train_ds, batch_size=best_params['batch_size'], shuffle=True, collate_fn=pad_collate_fn), DataLoader(final_test_ds, batch_size=best_params['batch_size'], shuffle=False, collate_fn=pad_collate_fn)

            final_model = GRUPredictor(1 + all_sample_features.shape[1] + 1, best_params['hidden_size'], best_params['num_layers'], 1)
            test_loss = train_and_evaluate(final_model, final_train_loader, final_test_loader, best_params)
            outer_results.append(test_loss)
            print(f"--- 外层循环 {i+1} 最终结果: 测试集 MSE = {test_loss:.6f}\n")

        # --- 4. 总结 ---
        print("\n################## 嵌套交叉验证全部完成 ##################")
        print(f"平均测试集MSE: {np.mean(outer_results):.6f} ± {np.std(outer_results):.6f}")
        best_trial_idx = np.argmin(outer_results)
        global_best_params = best_params_list[best_trial_idx]
        print(f"全局最优超参数 (来自第 {best_trial_idx+1} 轮): {global_best_params}")