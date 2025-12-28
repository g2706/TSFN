import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # 导入划分工具
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 步骤 1: 加载和解析您的XLSX数据 (与之前相同)
# ==============================================================================
def load_and_process_data(file_path):
    """从指定的XLSX文件中加载数据"""
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。")
        return None, None, None, None
    except ImportError:
        print("错误：缺少 'openpyxl' 库。请先运行 'pip install openpyxl'。")
        return None, None, None, None

    static_feature_columns = [
        'Polymer_MW', 'LA/GA', 'Polymer_size', 'Zeta_Potential',  'PDI', 'EE', 'Drug_MW', 'LogP', 'HBD',
        'HBA', 'HA', 'RB', 'PSA', 'NAR', 'FSP3'
    ]

    grouped = df.groupby('sample_id')

    all_features = []
    all_curves = []
    all_times = []
    sample_ids = []

    for sample_id, group in grouped:
        features = group[static_feature_columns].iloc[0].values
        all_features.append(features)

        times = group['time'].values
        curves = group['release_percentage'].values
        all_times.append(times)
        all_curves.append(curves)
        sample_ids.append(sample_id)

    print(f"数据加载完成！共找到 {len(all_features)} 个独立样本。")
    return np.array(all_features, dtype=np.float32), all_curves, all_times, sample_ids

# 其他函数和类定义 (Dataset, Model, Predict Function等与之前相同)
# ... [为简洁起见，这里省略与之前版本完全相同的代码，下面会把它们放回主程序中] ...
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
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x); return self.fc(out[:, -1, :])

def predict_full_curve_irregular(model, initial_releases, initial_times, static_features, future_times, seq_len):
    model.eval()
    predicted_releases = list(initial_releases)
    current_release_seq, current_time_seq = list(initial_releases), list(initial_times)
    all_times = np.concatenate([initial_times, future_times])
    with torch.no_grad():
        for i in range(len(future_times)):
            combined_input_seq = []
            combined_input_seq.append(np.concatenate(([current_release_seq[0]], static_features, [0])))
            for k in range(1, len(current_release_seq)):
                time_delta = current_time_seq[k] - current_time_seq[k-1]
                combined_input_seq.append(np.concatenate(([current_release_seq[k]], static_features, [time_delta])))
            input_tensor = torch.from_numpy(np.array([combined_input_seq])).float()
            next_release_pred = model(input_tensor).item()
            predicted_releases.append(next_release_pred)
            current_release_seq.pop(0); current_time_seq.pop(0)
            current_release_seq.append(next_release_pred)
            current_time_seq.append(all_times[seq_len + i])
    return np.array(predicted_releases)

# ==============================================================================
# 主程序：整合所有步骤
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 加载数据 ---
    XLSX_FILE_PATH = 'release_data/sup_4_release_data.xlsx'
    sample_features, release_curves, time_vectors, sample_ids = load_and_process_data(XLSX_FILE_PATH)

    if sample_features is not None:
        # --- 2. 按样本划分训练集和测试集 ---
        # 创建一个样本索引数组 [0, 1, 2, ..., n-1]
        num_samples = len(sample_ids)
        indices = np.arange(num_samples)

        # 使用 sklearn 的 train_test_split 来划分索引
        # test_size=0.2 表示 20% 的样本作为测试集
        # random_state 是一个种子，确保每次划分结果都一样，方便复现
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

        print(f"\n样本划分完成：")
        print(f"训练集样本数: {len(train_indices)}")
        print(f"测试集样本数: {len(test_indices)}")

        # 使用索引来切分数据集
        feat_train, feat_test = sample_features[train_indices], sample_features[test_indices]
        curves_train, curves_test = [release_curves[i] for i in train_indices], [release_curves[i] for i in test_indices]
        times_train, times_test = [time_vectors[i] for i in train_indices], [time_vectors[i] for i in test_indices]

        # --- 3. 数据缩放 ---
        # 重要：缩放器 (Scaler) 必须在训练集上进行 fit，然后应用到训练集和测试集上
        # 这样可以防止测试集的信息泄露到训练过程中
        feature_scaler = MinMaxScaler()
        release_scaler = MinMaxScaler()

        # 在训练数据上 fit
        scaled_feat_train = feature_scaler.fit_transform(feat_train)
        flat_releases_train = np.concatenate(curves_train).reshape(-1, 1)
        release_scaler.fit(flat_releases_train)

        # 在训练集和测试集上 transform
        scaled_curves_train = [release_scaler.transform(c.reshape(-1, 1)).flatten() for c in curves_train]
        scaled_feat_test = feature_scaler.transform(feat_test) # 使用训练集的scaler
        scaled_curves_test = [release_scaler.transform(c.reshape(-1, 1)).flatten() for c in curves_test]

        # --- 4. 创建Dataset和DataLoader ---
        SEQ_LENGTH = 3
        train_dataset = DrugReleaseDataset(scaled_feat_train, times_train, scaled_curves_train, SEQ_LENGTH)
        # 我们可以为测试集也创建一个Dataset，用于整体评估
        test_dataset = DrugReleaseDataset(scaled_feat_test, times_test, scaled_curves_test, SEQ_LENGTH)

        if len(train_dataset) == 0:
            print("错误：训练数据太少，无法创建任何训练序列。")
        else:
            train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
            test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate_fn)

            # --- 5. 初始化并训练模型 ---
            INPUT_SIZE = 1 + sample_features.shape[1] + 1
            model = GRUPredictor(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, output_size=1)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

            print("\n开始训练模型 (仅使用训练集)...")
            epochs = 300
            for epoch in range(epochs):
                model.train()
                for x_padded, y_true in train_loader:
                    y_pred = model(x_padded)
                    loss = criterion(y_pred, y_true.unsqueeze(1))
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                if (epoch + 1) % 50 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
            print("训练完成！\n")

            # --- 6. 在测试集上评估模型 ---
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for x_padded, y_true in test_loader:
                    y_pred = model(x_padded)
                    loss = criterion(y_pred, y_true.unsqueeze(1))
                    total_test_loss += loss.item() * x_padded.size(0) # 乘以batch size

            avg_test_loss = total_test_loss / len(test_dataset)
            print(f"在测试集上的平均均方误差 (MSE): {avg_test_loss:.6f}")

            # --- 7. 在一个 *未见过* 的测试样本上进行预测和可视化 ---
            print("\n在一个测试样本上运行预测并生成可视化图表...")
            test_sample_local_idx = 0 # 选择测试集中的第一个样本
            # 获取它在原始数据集中的真实索引
            original_sample_idx = test_indices[test_sample_local_idx]

            true_curve_original = release_curves[original_sample_idx]
            true_times_original = time_vectors[original_sample_idx]

            # 使用已经缩放好的测试数据
            scaled_release = scaled_curves_test[test_sample_local_idx]
            scaled_feature = scaled_feat_test[test_sample_local_idx]

            initial_releases_scaled = scaled_release[:SEQ_LENGTH]
            initial_times = true_times_original[:SEQ_LENGTH]
            future_times = true_times_original[SEQ_LENGTH:]

            predicted_curve_scaled = predict_full_curve_irregular(
                model, initial_releases_scaled, initial_times, scaled_feature, future_times, SEQ_LENGTH
            )

            predicted_curve_original = release_scaler.inverse_transform(predicted_curve_scaled.reshape(-1, 1)).flatten()

            plt.figure(figsize=(12, 7))
            plt.title(f'测试集样本 (原始ID: {sample_ids[original_sample_idx]}) 的释放曲线预测')
            plt.plot(true_times_original, true_curve_original, 'bo-', label='真实曲线 (Actual)')
            plt.plot(true_times_original, predicted_curve_original, 'ro--', label='预测曲线 (Predicted)')
            plt.plot(true_times_original[:SEQ_LENGTH], true_curve_original[:SEQ_LENGTH], 'go', markersize=10, label='初始输入点 (Initial)')
            plt.xlabel('时间 (Time)')
            plt.ylabel('释放百分比 (Release Percentage)')
            plt.legend()
            plt.grid(True)
            plt.show()