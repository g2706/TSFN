import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import higher

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
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- 步骤3: 数据准备 ---
class CurveTaskDataset(Dataset):
    # *** 关键恢复：__init__ 方法现在接收完整df，并在这里fit scaler ***
    def __init__(self, df, static_feature_columns, seq_len, k_shot):
        self.tasks = []
        self.feature_scaler = MinMaxScaler()
        self.release_scaler = MinMaxScaler()

        self.seq_len = seq_len

        # *** 恢复的数据泄露逻辑：在完整数据上fit缩放器 ***
        self.feature_scaler.fit(df[static_feature_columns])
        self.release_scaler.fit(df[['release_percentage']])

        for sample_id, group in df.groupby('sample_id'):
            num_sequences = len(group) - self.seq_len
            if num_sequences <= k_shot: continue

            features_s = self.feature_scaler.transform(group[static_feature_columns].iloc[[0]])[0]
            releases_s = self.release_scaler.transform(group[['release_percentage']]).flatten()
            times = group['time'].values

            task_sequences_X, task_sequences_y = [], []
            for j in range(num_sequences):
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
                'full_times': times,
                'full_releases_scaled': releases_s,
                'static_features_scaled': features_s,
                'sample_id': sample_id, 'drug_name': group['drug_name'].iloc[0]
            })

    def __len__(self): return len(self.tasks)
    def __getitem__(self, idx): return self.tasks[idx]

# --- 步骤4: MAML 训练器 ---
class MAMLTrainer:
    def __init__(self, model, meta_lr, fast_lr, num_updates):
        self.model = model
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        self.fast_lr = fast_lr
        self.num_updates = num_updates
        self.criterion = nn.L1Loss()

    def meta_train(self, dataset, meta_epochs, meta_batch_size, k_shot):
        for epoch in range(meta_epochs):
            self.model.train()
            task_batch_indices = np.random.choice(len(dataset), meta_batch_size, replace=False)
            task_batch = [dataset[i] for i in task_batch_indices]
            self.meta_optimizer.zero_grad()
            total_query_loss = 0.0
            for task in task_batch:
                X_task, y_task = task['X'].to(device), task['y'].to(device)
                X_support, y_support = X_task[:k_shot], y_task[:k_shot]
                X_query, y_query = X_task[k_shot:], y_task[k_shot:]
                if not len(X_query): continue
                with higher.innerloop_ctx(self.model, self.meta_optimizer, copy_initial_weights=True) as (fmodel, diffopt):
                    for _ in range(self.num_updates):
                        support_pred = fmodel(X_support)
                        support_loss = self.criterion(support_pred, y_support)
                        diffopt.step(support_loss)
                    query_pred = fmodel(X_query)
                    query_loss = self.criterion(query_pred, y_query)
                    query_loss.backward()
                    total_query_loss += query_loss.item()
            self.meta_optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Meta Epoch [{epoch+1}/{meta_epochs}], Avg Query Loss: {total_query_loss / meta_batch_size:.6f}")

# --- 主程序 ---
if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"主程序运行在 '{device}' 设备上。")

    # ============================ 调优配置区 ============================
    META_LR = 1e-3
    FAST_LR = 0.001
    NUM_UPDATES = 5
    META_BATCH_SIZE = 10
    SEQ_LEN = 2
    NUM_ADAPTATION_POINTS = 6
    MODEL_HIDDEN_SIZE = 64
    MODEL_NUM_LAYERS = 3
    NUM_ADAPTATION_STEPS = 20
    META_EPOCHS = 200
    # ===================================================================

    K_SHOT = NUM_ADAPTATION_POINTS - SEQ_LEN
    print(f"配置: GRU输入长度(SEQ_LEN)={SEQ_LEN}, 适应点数(NUM_ADAPTATION_POINTS)={NUM_ADAPTATION_POINTS}")
    print(f"  => 每个任务将使用 {K_SHOT} 个样本进行快速适应 (k_shot)。")

    # --- 数据加载和准备 (恢复为包含数据泄露的版本) ---
    df = pd.read_excel("release_data/sup_4_release_data.xlsx")
    df['drug_name'] = df['drug_name'].str.strip().str.lower()
    static_cols = ['Polymer_MW', 'LA/GA', 'Polymer_size', 'Zeta_Potential', 'PDI', 'EE', 'Drug_MW', 'LogP', 'HBD', 'HBA', 'HA', 'RB', 'PSA', 'NAR', 'FSP3']

    # 1. 直接用完整df创建数据集，Scaler将在CurveTaskDataset内部被fit
    full_dataset = CurveTaskDataset(df, static_cols, SEQ_LEN, K_SHOT)
    print(f"数据加载完成，过滤后剩余 {len(full_dataset)} 个有效任务。")

    # 2. 划分任务索引用于元训练和元测试
    tasks = list(range(len(full_dataset)))
    random.shuffle(tasks)
    train_task_indices = tasks[:int(0.8 * len(tasks))]
    test_task_indices = tasks[int(0.8 * len(tasks)):]

    meta_train_dataset = [full_dataset[i] for i in train_task_indices]
    meta_test_dataset = [full_dataset[i] for i in test_task_indices]

    # --- 模型和训练器初始化 ---
    input_size = 1 + len(static_cols) + 1
    meta_model = GRUPredictor(input_size=input_size, hidden_size=MODEL_HIDDEN_SIZE, num_layers=MODEL_NUM_LAYERS).to(device)
    trainer = MAMLTrainer(model=meta_model, meta_lr=META_LR, fast_lr=FAST_LR, num_updates=NUM_UPDATES)

    # --- 元训练 ---
    print("\n--- 开始元训练 ---")
    trainer.meta_train(
        dataset=meta_train_dataset,
        meta_epochs=META_EPOCHS,
        meta_batch_size=META_BATCH_SIZE,
        k_shot=K_SHOT
    )
    print("--- 元训练完成 ---")

    # --- 在一个未见过的任务上评估 ---
    print("\n--- 在一个测试任务上评估元学习效果 ---")
    test_task = meta_test_dataset[0]

    X_support, y_support = test_task['X'][:K_SHOT].to(device), test_task['y'][:K_SHOT].to(device)
    adapted_model = GRUPredictor(input_size=input_size, hidden_size=MODEL_HIDDEN_SIZE, num_layers=MODEL_NUM_LAYERS).to(device)
    adapted_model.load_state_dict(meta_model.state_dict())
    adaptation_optimizer = optim.SGD(adapted_model.parameters(), lr=trainer.fast_lr)
    for _ in range(NUM_ADAPTATION_STEPS):
        pred = adapted_model(X_support)
        loss = trainer.criterion(pred, y_support)
        adaptation_optimizer.zero_grad(); loss.backward(); adaptation_optimizer.step()
    print(f"对新任务 '{test_task['drug_name']}' (ID: {test_task['sample_id']}) 适应完成。")

    adapted_model.eval()
    with torch.no_grad():
        full_times = test_task['full_times']; full_releases_s = test_task['full_releases_scaled']; static_features_s = test_task['static_features_scaled']
        predictions_s = list(full_releases_s[:NUM_ADAPTATION_POINTS])
        current_release_seq_s = list(full_releases_s[NUM_ADAPTATION_POINTS - SEQ_LEN : NUM_ADAPTATION_POINTS])
        for i in range(NUM_ADAPTATION_POINTS - 1, len(full_times) - 1):
            combined_seq = []
            for k in range(SEQ_LEN):
                time_delta = full_times[i - (SEQ_LEN-1) + k + 1] - full_times[i - (SEQ_LEN-1) + k]
                step_input = np.concatenate(([current_release_seq_s[k]], static_features_s, [time_delta]))
                combined_seq.append(step_input)
            input_tensor = torch.from_numpy(np.array([combined_seq])).float().to(device)
            next_pred_s = adapted_model(input_tensor).cpu().numpy().flatten()[0]
            predictions_s.append(next_pred_s)
            current_release_seq_s.pop(0); current_release_seq_s.append(next_pred_s)

    true_curve_raw = full_dataset.release_scaler.inverse_transform(full_releases_s.reshape(-1, 1))
    pred_curve_raw = full_dataset.release_scaler.inverse_transform(np.array(predictions_s).reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(full_times, true_curve_raw, 'bo-', label='Experimental Curve')
    plt.plot(full_times, pred_curve_raw, 'ro--', label='Meta-Learned Prediction')
    plt.axvspan(0, full_times[NUM_ADAPTATION_POINTS-1], color='gray', alpha=0.2, label=f'Adaptation Data (First {NUM_ADAPTATION_POINTS} points)')
    plt.title(f"Meta-Learning Prediction for {test_task['drug_name']}")
    plt.legend()
    plt.show()