import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler # <--- 修改: 引入StandardScaler
from sklearn.metrics import mean_absolute_error
from skorch import NeuralNetRegressor
from matplotlib import pyplot as plt

class IdentityScaler:
    """一个什么都不做的伪归一化器，用于保持代码结构不变。"""
    def fit(self, data):
        return self
    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            return data.values
        return data
    def inverse_transform(self, data):
        return data

# --- 确保模型定义与训练时一致 ---
class GRUFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(GRUFeatureExtractor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.to(torch.float32)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# PyTorch的安全列表，确保自定义类可以被加载
torch.serialization.add_safe_globals([
    GRUFeatureExtractor,
    torch.nn.GRU,
    torch.nn.Linear,
    torch.nn.MSELoss,
    torch.optim.Adam,
])


# --- 核心预测函数 ---
def predict_new_data(new_data_df, model_pipeline, config):
    """
    使用加载的模型管道对新数据进行自回归预测。
    """

    # <--- 修改: 从管道中解包新增的 time_delta_scaler
    gru_extractor = model_pipeline['gru_extractor']
    lgbm_predictor = model_pipeline['lgbm_predictor']
    feature_scaler = model_pipeline['feature_scaler']
    release_scaler = model_pipeline['release_scaler']
    time_delta_scaler = model_pipeline['time_delta_scaler'] # 新增
    lgbm_feature_names = model_pipeline['lgbm_feature_names']

    seq_len = config['seq_len']
    num_prefix_points = config['num_prefix_points']
    static_feature_columns = config['static_features']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = []

    for sid, sample_data in new_data_df.groupby('sample_id'):
        print(f"  > 正在预测样本 ID: {sid}...")
        if len(sample_data) < num_prefix_points:
            print(f"    > 警告: 样本 {sid} 数据点过少 ({len(sample_data)}个)，无法进行预测，已跳过。")
            continue

        static_features = sample_data[static_feature_columns].iloc[0].values
        static_features_s = feature_scaler.transform(sample_data[static_feature_columns].iloc[[0]])[0]
        times = sample_data['time'].values
        true_releases_s = release_scaler.transform(sample_data[['release_percentage']]).flatten()
        predicted_releases_s = list(true_releases_s[:num_prefix_points])

        gru_extractor.module_.eval()
        with torch.no_grad():
            for i in range(num_prefix_points, len(times)):
                history_seq_release = predicted_releases_s[i-seq_len : i]
                history_seq_times = times[i-seq_len : i]

                if np.isnan(history_seq_release).any():
                    print(f"    > 警告: 样本 {sid} 在时间步 {i} 遇到无效历史数据，停止预测。")
                    last_valid_pred = predicted_releases_s[-1]
                    predicted_releases_s.extend([last_valid_pred] * (len(times) - len(predicted_releases_s)))
                    break

                gru_input = []
                for k in range(seq_len):
                    time_delta = (history_seq_times[k] - history_seq_times[k-1]) if k > 0 else 0
                    # <--- 修改: 使用加载的 scaler 对 time_delta 进行标准化
                    time_delta_s = time_delta_scaler.transform([[time_delta]])[0, 0]
                    gru_input.append(np.concatenate(([history_seq_release[k]], static_features_s, [time_delta_s])))

                input_tensor = torch.from_numpy(np.array([gru_input])).float().to(device)
                gru_hidden_state, _ = gru_extractor.module_.gru(input_tensor)
                sequence_feature = gru_hidden_state.cpu().numpy()[:, -1, :].flatten()
                time_feature = times[i]
                lgbm_input_data = np.concatenate([static_features, [time_feature], sequence_feature]).reshape(1, -1)
                lgbm_input_df = pd.DataFrame(lgbm_input_data, columns=lgbm_feature_names)
                next_pred_s = lgbm_predictor.predict(lgbm_input_df)[0]
                predicted_releases_s.append(next_pred_s)

                # <--- 修改: 移除了重复的打印和预测代码，让逻辑更清晰
                # print(f"Time: {times[i]}, GRU Feature (前5个值): {sequence_feature[:5]}")
                # print(f"LGBM Prediction: {next_pred_s}")

        true_curve_raw = release_scaler.inverse_transform(true_releases_s.reshape(-1, 1)).flatten()
        predicted_curve_raw = release_scaler.inverse_transform(np.array(predicted_releases_s).reshape(-1, 1)).flatten()

        for k in range(len(times)):
            record = {'Experimental Index': sid, 'DP_Groups': sample_data['drug_name'].iloc[0], 'Time': times[k],
                      'Experimental_Release': true_curve_raw[k], 'Predicted_Release': predicted_curve_raw[k]}
            all_results.append(record)

    return pd.DataFrame(all_results)

if __name__ == '__main__':
    # ============================ 配置区域 ============================
    MODEL_FILE_PATH = "release_Trained_models_hyb/Hybrid_Autoregressive_Model_1_shot/Hybrid_Autoregressive_Model_1_shot_final_model.pkl"
    NEW_DATA_FILEPATH = "release_data/sup_real_release_data.xlsx"
    NUM_PREFIX_POINTS = 2
    SEQ_LEN=2
    STATIC_FEATURES = ['Polymer_MW', 'LA/GA', 'Polymer_size', 'Zeta_Potential', 'PDI', 'EE', 'Drug_MW', 'LogP', 'HBD', 'HBA', 'HA', 'RB', 'PSA', 'NAR', 'FSP3']
    OUTPUT_FIGURE_DIR = "release_Figures_NEW_PREDICTIONS"
    # ================================================================

    print("--- 开始加载模型管道 ---")
    try:
        with open(MODEL_FILE_PATH, 'rb') as f:
            model_pipeline = pickle.load(f)
        # <--- 修改: 验证最终模型是否包含新的 time_delta_scaler
        required_keys = ['gru_extractor', 'lgbm_predictor', 'feature_scaler', 'release_scaler', 'time_delta_scaler', 'lgbm_feature_names']
        if not all(key in model_pipeline for key in required_keys):
            raise ValueError("模型文件.pkl中缺少必要的组件！请确保您加载的是最新训练的模型。")
        print("模型管道加载成功！")
        # ==================== 新增的验证代码 ====================
        print("\n--- 模型管道内容验证 ---")
        print("已加载的组件:", list(model_pipeline.keys()))

        # 打印已加载的GRU模型的超参数
        # skorch对象有.get_params()方法可以查看所有参数
        print("\n已加载GRU模型的参数:")
        gru_params = model_pipeline['gru_extractor'].get_params()
        print(f"  - 学习率 (lr): {gru_params.get('lr')}")
        print(f"  - 隐藏层大小 (hidden_size): {gru_params.get('module__hidden_size')}")
        print(f"  - 网络层数 (num_layers): {gru_params.get('module__num_layers')}")

        # 打印已加载的LGBM模型的超参数
        print("\n已加载LGBM模型的参数:")
        print(model_pipeline['lgbm_predictor'].get_params())
        print("------------------------\n")
        # =======================================================
    except FileNotFoundError as e:
        print(f"错误：找不到模型文件！请检查 `MODEL_FILE_PATH` 是否正确。\n具体文件: {e.filename}"); exit()
    except Exception as e:
        print(f"加载模型时发生错误: {e}"); exit()

    print("\n--- 开始加载新数据 ---")
    try:
        new_df = pd.read_excel(NEW_DATA_FILEPATH)
        print(f"成功加载 {len(new_df['sample_id'].unique())} 个新样本。")
    except FileNotFoundError as e:
        print(f"错误：找不到新数据文件！请检查 `NEW_DATA_FILEPATH` 是否正确。\n路径: {e.filename}"); exit()

    # 动态推断seq_len
    # 假设GRU输入为 [release (1), static_features (N), time_delta (1)]
    gru_input_size = model_pipeline['gru_extractor'].module_.gru.input_size
    seq_len_from_model = 4

    config_pred = {
        'seq_len': SEQ_LEN,
        'num_prefix_points': NUM_PREFIX_POINTS,
        'static_features': STATIC_FEATURES
    }

    # 增加一个检查，确保 num_prefix_points 足够
    if config_pred['num_prefix_points'] < config_pred['seq_len']:
        raise ValueError(f"错误: NUM_PREFIX_POINTS (值为{config_pred['num_prefix_points']}) 不能小于模型训练时的序列长度 seq_len (值为{config_pred['seq_len']})")


    results_df = predict_new_data(new_df, model_pipeline, config_pred)

    if results_df.empty:
        print("\n未能生成任何预测结果。程序终止。"); exit()

    results_df_eval = results_df.dropna()
    overall_mae = mean_absolute_error(results_df_eval['Experimental_Release'], results_df_eval['Predicted_Release'])
    print(f"\n--- 预测完成 ---")
    print(f"在新数据上的总体平均绝对误差 (MAE): {overall_mae:.6f}")

    os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)
    print(f"\n开始为 {len(results_df['Experimental Index'].unique())} 个新样本生成对比图...")

    for name, group in results_df.groupby('Experimental Index'):
        group = group.sort_values(by='Time')
        X = group['Time']
        Y_pred = group['Predicted_Release']
        Y_exp = group['Experimental_Release']
        dp = group['DP_Groups'].iloc[0]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(X, Y_pred, label='Predicted (Hybrid Model)', linestyle='--', marker='o', markersize=8, markeredgecolor="black", alpha=0.8)
        ax.plot(X, Y_exp, label='Experimental', linestyle='--', marker='o', markersize=8, alpha=0.8, markeredgecolor="black")

        # 确保有足够的前缀点来绘制灰色区域
        if len(group) >= NUM_PREFIX_POINTS:
            ax.axvspan(0, group['Time'].iloc[NUM_PREFIX_POINTS-1], color='gray', alpha=0.2, label=f'Known Prefix (First {NUM_PREFIX_POINTS} points)')

        ax.set(ylim=(-0.02, 1.08), yticks=np.arange(0, 1.08, 0.1))
        ax.set_xlabel('Time (Hours)', fontsize=15, color='black', weight='bold')
        ax.set_ylabel('Fractional Drug Release', fontsize=15, color='black', weight='bold')
        ax.set_title(f'{dp} (Sample ID {name})', color='black', weight='bold', fontsize=18, pad=20)
        ax.legend(loc='upper left', frameon=False, prop={'size': 12})
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, f'Prediction_SampleID_{name}.png'), dpi=600)
        plt.close(fig)

    print(f"绘图完成！图片已保存在 '{OUTPUT_FIGURE_DIR}' 文件夹中。")