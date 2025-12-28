import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import higher  # 导入 higher 库

# --- 步骤1: 实现可复现性 ---
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为: {seed}")

# --- 步骤2: 模型定义 ---
class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(GRUPredictor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.to(torch.float32)
        # *** 关键修复：在这里禁用cuDNN ***
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- 步骤3: 数据准备 ---
class CurveTaskDataset(Dataset):
    # ... (与上一版完全相同) ...
    def __init__(self, df, static_feature_columns, seq_len=1):
        self.tasks = []
        self.feature_scaler = MinMaxScaler()
        self.release_scaler = MinMaxScaler()
        self.feature_scaler.fit(df[static_feature_columns])
        self.release_scaler.fit(df[['release_percentage']])
        for sample_id, group in df.groupby('sample_id'):
            if len(group) <= seq_len + 1: continue
            features_s = self.feature_scaler.transform(group[static_feature_columns].iloc[[0]])[0]
            releases_s = self.release_scaler.transform(group[['release_percentage']]).flatten()
            times = group['time'].values
            task_sequences_X, task_sequences_y = [], []
            for j in range(len(releases_s) - seq_len):
                seq_in_release = releases_s[j:j+seq_len]
                seq_out_release = releases_s[j+seq_len]
                combined_seq = []
                for k in range(seq_len):
                    time_delta = times[j+k+1] - times[j+k]
                    combined_seq.append(np.concatenate(([seq_in_release[k]], features_s, [time_delta])))
                task_sequences_X.append(np.array(combined_seq))
                task_sequences_y.append(seq_out_release)
            self.tasks.append({
                'X': torch.tensor(np.array(task_sequences_X), dtype=torch.float32),
                'y': torch.tensor(np.array(task_sequences_y), dtype=torch.float32).view(-1, 1),
                'sample_id': sample_id, 'drug_name': group['drug_name'].iloc[0]
            })
    def __len__(self): return len(self.tasks)
    def __getitem__(self, idx): return self.tasks[idx]

# --- 步骤4: MAML 训练器 ---
class MAMLTrainer:
    # ... (__init__ 与上一版完全相同) ...
    def __init__(self, model, meta_lr, fast_lr, num_updates):
        self.model = model
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        self.fast_lr = fast_lr
        self.num_updates = num_updates
        self.criterion = nn.L1Loss()

    # *** meta_train 方法不再需要修改，因为修改已移入模型forward函数中 ***
    def meta_train(self, dataset, meta_epochs, meta_batch_size, k_shot):
        for epoch in range(meta_epochs):
            self.model.train()
            task_batch_indices = np.random.choice(len(dataset), meta_batch_size, replace=False)
            task_batch = [dataset[i] for i in task_batch_indices]
            self.meta_optimizer.zero_grad()

            for task in task_batch:
                X_task, y_task = task['X'].to(device), task['y'].to(device)
                support_indices = np.random.choice(len(X_task), k_shot, replace=False)
                query_indices = [i for i in range(len(X_task)) if i not in support_indices]
                X_support, y_support = X_task[support_indices], y_task[support_indices]
                X_query, y_query = X_task[query_indices], y_task[query_indices]

                with higher.innerloop_ctx(self.model, self.meta_optimizer, copy_initial_weights=True) as (fmodel, diffopt):
                    for _ in range(self.num_updates):
                        support_pred = fmodel(X_support)
                        support_loss = self.criterion(support_pred, y_support)
                        diffopt.step(support_loss)

                    query_pred = fmodel(X_query)
                    query_loss = self.criterion(query_pred, y_query)
                    query_loss.backward()

            self.meta_optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Meta Epoch [{epoch+1}/{meta_epochs}], Last Task Query Loss: {query_loss.item():.6f}")

# --- 主程序 ---
if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"主程序运行在 '{device}' 设备上。")

    # --- 数据加载和准备 ---
    df = pd.read_excel("release_data/sup_4_release_data.xlsx")
    df['drug_name'] = df['drug_name'].str.strip().str.lower()
    static_cols = ['Polymer_MW', 'LA/GA', 'Polymer_size', 'Zeta_Potential', 'PDI', 'EE', 'Drug_MW', 'LogP', 'HBD', 'HBA', 'HA', 'RB', 'PSA', 'NAR', 'FSP3']

    full_dataset = CurveTaskDataset(df, static_cols)

    tasks = list(range(len(full_dataset)))
    random.shuffle(tasks)
    train_task_indices = tasks[:int(0.8 * len(tasks))]
    test_task_indices = tasks[int(0.8 * len(tasks)):]

    # --- 模型和训练器初始化 ---
    input_size = full_dataset[0]['X'].shape[2]
    meta_model = GRUPredictor(input_size=input_size).to(device)

    trainer = MAMLTrainer(model=meta_model, meta_lr=1e-3, fast_lr=0.01, num_updates=1)

    # --- 元训练 ---
    print("\n--- 开始元训练 ---")
    trainer.meta_train(
        dataset=[full_dataset[i] for i in train_task_indices],
        meta_epochs=100,
        meta_batch_size=5,
        k_shot=4
    )
    print("--- 元训练完成 ---")

    # --- 在一个未见过的任务上评估 ---
    print("\n--- 在一个测试任务上评估元学习效果 ---")
    test_task = full_dataset[test_task_indices[0]]
    X_test_task, y_test_task = test_task['X'], test_task['y']

    k_shot_eval = 5
    X_support, y_support = X_test_task[:k_shot_eval].to(device), y_test_task[:k_shot_eval].to(device)

    adapted_model = GRUPredictor(input_size=input_size).to(device)
    adapted_model.load_state_dict(meta_model.state_dict())
    adaptation_optimizer = optim.SGD(adapted_model.parameters(), lr=trainer.fast_lr)

    # 常规微调，不需要禁用cuDNN
    for _ in range(10):
        pred = adapted_model(X_support)
        loss = trainer.criterion(pred, y_support)
        adaptation_optimizer.zero_grad()
        loss.backward()
        adaptation_optimizer.step()

    print(f"对新任务 '{test_task['drug_name']}' (ID: {test_task['sample_id']}) 适应完成。")

    adapted_model.eval()
    with torch.no_grad():
        full_pred_scaled = adapted_model(X_test_task.to(device)).cpu().numpy()

    true_curve_scaled = y_test_task.numpy()
    true_curve_raw = full_dataset.release_scaler.inverse_transform(true_curve_scaled)
    pred_curve_raw = full_dataset.release_scaler.inverse_transform(full_pred_scaled)

    plt.figure(figsize=(10, 6))
    plt.plot(true_curve_raw, 'bo-', label='Experimental Curve')
    plt.plot(pred_curve_raw, 'ro--', label='Meta-Learned Prediction')
    plt.title(f"Meta-Learning Prediction for {test_task['drug_name']}")
    plt.legend()
    plt.show()