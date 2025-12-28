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
# 未使用合适的时间差数据




# 确保安装了openpyxl: pip install openpyxl
# ==============================================================================
# 步骤 1 & 2: 数据加载和模型/数据类定义 (与之前类似)
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

class DrugReleaseDataset(Dataset):
    def __init__(self, features, times, releases, sequence_length=3):
        self.X, self.y = [], []
        for i in range(len(features)):
            sample_feat, sample_time, sample_release = features[i], times[i], releases[i]
            if len(sample_release) > sequence_length:
                for j in range(len(sample_release) - sequence_length):
                    seq_in_release, seq_in_time = sample_release[j:j+sequence_length], sample_time[j:j+sequence_length]
                    seq_out_release = sample_release[j+sequence_length]
                    combined_seq = []
                    combined_seq.append(np.concatenate(([seq_in_release[0]], sample_feat, [0])))
                    for k in range(1, sequence_length):
                        time_delta = seq_in_time[k] - seq_in_time[k-1]
                        combined_seq.append(np.concatenate(([seq_in_release[k]], sample_feat, [time_delta])))
                    self.X.append(np.array(combined_seq))
                    self.y.append(seq_out_release)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).float()

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
# 步骤 3: 核心训练/评估函数
# ==============================================================================
def train_and_evaluate(model, train_loader, val_loader, params):
    """在一个训练集上训练，并在一个验证集上评估"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # 训练循环
    model.train()
    for epoch in range(params['epochs']):
        for x_padded, y_true in train_loader:
            y_pred = model(x_padded) # 正确使用 x_padded 作为输入
            loss = criterion(y_pred, y_true.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 评估循环
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x_padded, y_true in val_loader:
            # *** 这是被修复的行 ***
            y_pred = model(x_padded) # 应该使用验证集的输入 x_padded
            loss = criterion(y_pred, y_true.unsqueeze(1))
            total_val_loss += loss.item() * x_padded.size(0)

    # 确保分母不为0
    if len(val_loader.dataset) == 0:
        return float('inf') # 如果验证集为空，返回一个极大值

    return total_val_loss / len(val_loader.dataset)
# ==============================================================================
# 主程序：实现嵌套交叉验证
# ==============================================================================
if __name__ == '__main__':
    # --- 全局参数 ---
    XLSX_FILE_PATH = 'release_data/sup_4_release_data.xlsx'
    SEQ_LENGTH = 3
    NUM_TRIALS = 5  # 外层循环次数 (对应您代码的NUM_TRIALS)
    N_ITER_SEARCH = 10 # 内层随机搜索的超参数组合数量 (对应您代码的n_iter)
    K_FOLDS_INNER = 5  # 内层交叉验证的折数

    # --- 1. 加载数据 ---
    all_sample_features, all_release_curves, all_time_vectors, all_sample_ids = load_and_process_data(XLSX_FILE_PATH)

    if all_sample_features is not None:
        # --- 2. 定义GRU的超参数搜索空间 (对应您的p_grid) ---
        p_grid = {
            'hidden_size': [32, 64, 128],
            'num_layers': [1, 2, 3],
            'lr': [0.01, 0.005, 0.001],
            'batch_size': [16, 32],
            'epochs': [100] # 为加速演示，固定epoch，实际可加入搜索
        }

        outer_results = []
        best_params_list = []

        # --- 3. 外层交叉验证循环 ---
        print(f"--- 开始嵌套交叉验证 (共 {NUM_TRIALS} 轮外层循环) ---\n")
        sample_indices = np.arange(len(all_sample_ids))

        for i in range(NUM_TRIALS):
            print(f"\n################## 外层循环 {i+1}/{NUM_TRIALS} ##################")

            # 3.1. 按样本划分外层的训练集和测试集
            outer_train_indices, outer_test_indices = train_test_split(sample_indices, test_size=0.2, random_state=i)

            # 3.2. 数据缩放 (Scaler在每个外层循环的训练集上重新fit)
            feature_scaler = MinMaxScaler()
            release_scaler = MinMaxScaler()

            feat_outer_train = all_sample_features[outer_train_indices]
            curves_outer_train = [all_release_curves[j] for j in outer_train_indices]

            scaled_feat_outer_train = feature_scaler.fit_transform(feat_outer_train)
            release_scaler.fit(np.concatenate(curves_outer_train).reshape(-1, 1))

            # --- 3.3. 内层循环：超参数搜索 ---
            print(f"--- 开始内层 {K_FOLDS_INNER}-折交叉验证以搜索超参数 ({N_ITER_SEARCH} 次尝试) ---")
            hyperparam_search_results = []

            for j in range(N_ITER_SEARCH):
                # 3.3.1. 随机选择一组超参数
                params = {k: random.choice(v) for k, v in p_grid.items()}

                # 3.3.2. 在外层训练集上进行K-Fold交叉验证
                kf_inner = KFold(n_splits=K_FOLDS_INNER, shuffle=True, random_state=j)
                inner_fold_losses = []

                for inner_train_idx_of_idx, inner_val_idx_of_idx in kf_inner.split(outer_train_indices):
                    # 获取真实的样本索引
                    inner_train_indices = outer_train_indices[inner_train_idx_of_idx]
                    inner_val_indices = outer_train_indices[inner_val_idx_of_idx]

                    # 准备当前折的数据和缩放
                    # ... [此处省略了详细的数据准备代码，逻辑与之前版本类似] ...
                    feat_inner_train = all_sample_features[inner_train_indices]
                    curves_inner_train_raw = [all_release_curves[k] for k in inner_train_indices]
                    times_inner_train = [all_time_vectors[k] for k in inner_train_indices]

                    feat_inner_val = all_sample_features[inner_val_indices]
                    curves_inner_val_raw = [all_release_curves[k] for k in inner_val_indices]
                    times_inner_val = [all_time_vectors[k] for k in inner_val_indices]

                    # 使用外层循环的scaler来转换数据
                    scaled_feat_inner_train = feature_scaler.transform(feat_inner_train)
                    scaled_curves_inner_train = [release_scaler.transform(c.reshape(-1,1)).flatten() for c in curves_inner_train_raw]
                    scaled_feat_inner_val = feature_scaler.transform(feat_inner_val)
                    scaled_curves_inner_val = [release_scaler.transform(c.reshape(-1,1)).flatten() for c in curves_inner_val_raw]

                    # 创建Dataset和DataLoader
                    train_ds_inner = DrugReleaseDataset(scaled_feat_inner_train, times_inner_train, scaled_curves_inner_train, SEQ_LENGTH)
                    val_ds_inner = DrugReleaseDataset(scaled_feat_inner_val, times_inner_val, scaled_curves_inner_val, SEQ_LENGTH)

                    if len(train_ds_inner) == 0 or len(val_ds_inner) == 0: continue # 如果数据太少，跳过

                    train_loader_inner = DataLoader(train_ds_inner, batch_size=params['batch_size'], shuffle=True, collate_fn=pad_collate_fn)
                    val_loader_inner = DataLoader(val_ds_inner, batch_size=params['batch_size'], shuffle=False, collate_fn=pad_collate_fn)

                    # 初始化模型并训练评估
                    INPUT_SIZE = 1 + all_sample_features.shape[1] + 1
                    model = GRUPredictor(INPUT_SIZE, params['hidden_size'], params['num_layers'], 1)

                    fold_loss = train_and_evaluate(model, train_loader_inner, val_loader_inner, params)
                    inner_fold_losses.append(fold_loss)

                # 计算这组超参数的平均验证损失
                avg_val_loss = np.mean(inner_fold_losses)
                hyperparam_search_results.append({'params': params, 'loss': avg_val_loss})
                print(f"  > 尝试 {j+1}/{N_ITER_SEARCH}: params={params}, avg_val_loss={avg_val_loss:.6f}")

            # 3.4. 选出内层循环中最好的超参数
            best_inner_result = min(hyperparam_search_results, key=lambda x: x['loss'])
            best_params = best_inner_result['params']
            best_params_list.append(best_params)
            print(f"\n--- 外层循环 {i+1} 的最佳超参数已找到 ---\n  - 参数: {best_params}\n  - 最佳验证集MSE: {best_inner_result['loss']:.6f}")

            # --- 3.5. 在完整的外层训练集上训练，并在外层测试集上评估 ---
            print("--- 使用最佳参数在完整外层训练集上训练，并在外层测试集上评估 ---")
            # 准备数据
            # ... [此处省略数据准备代码，逻辑与之前版本类似] ...
            times_outer_train = [all_time_vectors[j] for j in outer_train_indices]
            scaled_curves_outer_train = [release_scaler.transform(c.reshape(-1,1)).flatten() for c in curves_outer_train]

            feat_outer_test = all_sample_features[outer_test_indices]
            curves_outer_test_raw = [all_release_curves[j] for j in outer_test_indices]
            times_outer_test = [all_time_vectors[j] for j in outer_test_indices]

            scaled_feat_outer_test = feature_scaler.transform(feat_outer_test)
            scaled_curves_outer_test = [release_scaler.transform(c.reshape(-1,1)).flatten() for c in curves_outer_test_raw]

            # 创建DataLoader
            final_train_ds = DrugReleaseDataset(scaled_feat_outer_train, times_outer_train, scaled_curves_outer_train, SEQ_LENGTH)
            final_test_ds = DrugReleaseDataset(scaled_feat_outer_test, times_outer_test, scaled_curves_outer_test, SEQ_LENGTH)

            final_train_loader = DataLoader(final_train_ds, batch_size=best_params['batch_size'], shuffle=True, collate_fn=pad_collate_fn)
            final_test_loader = DataLoader(final_test_ds, batch_size=best_params['batch_size'], shuffle=False, collate_fn=pad_collate_fn)

            # 训练和评估
            INPUT_SIZE = 1 + all_sample_features.shape[1] + 1
            final_model = GRUPredictor(INPUT_SIZE, best_params['hidden_size'], best_params['num_layers'], 1)
            test_loss = train_and_evaluate(final_model, final_train_loader, final_test_loader, best_params)

            outer_results.append(test_loss)
            print(f"\n--- 外层循环 {i+1} 最终结果 ---\n  - 测试集 MSE: {test_loss:.6f}\n")

        # --- 4. 总结所有结果 ---
        print("\n################## 嵌套交叉验证全部完成 ##################")
        print(f"在 {NUM_TRIALS} 轮外层交叉验证中的平均测试集MSE: {np.mean(outer_results):.6f} ± {np.std(outer_results):.6f}")

        # 选出全局最优模型 (例如，基于最低的测试MSE)
        best_trial_idx = np.argmin(outer_results)
        global_best_params = best_params_list[best_trial_idx]
        print(f"\n表现最好的一轮是第 {best_trial_idx+1} 轮，其测试MSE为: {outer_results[best_trial_idx]:.6f}")
        print(f"全局最优超参数为: {global_best_params}")

        # 你可以在这里使用 global_best_params 在全部数据上训练一个最终模型用于部署