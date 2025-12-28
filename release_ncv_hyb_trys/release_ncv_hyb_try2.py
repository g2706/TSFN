import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV as RSCV, GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from skorch import NeuralNetRegressor
import joblib

# --- 实现可复现性 ---
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为: {seed}")

# --- 模型定义 ---
class GRUFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(GRUFeatureExtractor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.to(torch.float32)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- 混合自回归模型框架 ---
class HybridAutoregressiveCV:
    def __init__(self, datafile, static_feature_columns, seq_len=3):
        self.df = pd.read_excel(datafile)
        self.df['drug_name'] = self.df['drug_name'].str.strip().str.lower()
        self.SEQ_LENGTH = seq_len
        self.static_feature_columns = static_feature_columns
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1)
        self.lgbm_p_grid = {
            'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [16, 31, 64], 'max_depth': [-1, 10],
        }
        self.lgbm_feature_names = None

        # 初始化用于存储结果的列表
        self.itr_number, self.outer_results, self.inner_results, self.model_params = [], [], [], []
        self.G_test_list, self.y_test_list, self.pred_list, self.E_test_list, self.T_test_list = [], [], [], [], []

    def _prepare_gru_teacher_forcing_data(self, sample_ids, feature_scaler, release_scaler):
        X_list, y_list = [], []
        for sid in sample_ids:
            group = self.df[self.df['sample_id'] == sid]
            features_s = feature_scaler.transform(group[self.static_feature_columns])[0]
            releases_s = release_scaler.transform(group[['release_percentage']]).flatten()
            times = group['time'].values
            if len(releases_s) > self.SEQ_LENGTH:
                for j in range(len(releases_s) - self.SEQ_LENGTH):
                    seq_in_release = releases_s[j:j+self.SEQ_LENGTH]
                    combined_seq = []
                    for k in range(self.SEQ_LENGTH):
                        time_delta = times[j+k+1] - times[j+k]
                        combined_seq.append(np.concatenate(([seq_in_release[k]], features_s, [time_delta])))
                    X_list.append(np.array(combined_seq))
                    y_list.append(releases_s[j+self.SEQ_LENGTH])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1)

    def _create_lgbm_training_set(self, gru_extractor, sample_ids, feature_scaler, release_scaler):
        all_lgbm_samples = []
        gru_extractor.module_.eval()
        with torch.no_grad():
            for sid in sample_ids:
                group = self.df[self.df['sample_id'] == sid].copy()
                if len(group) <= self.SEQ_LENGTH: continue
                features_s = feature_scaler.transform(group[self.static_feature_columns])[0]
                releases_s = release_scaler.transform(group[['release_percentage']]).flatten()
                times = group['time'].values
                for j in range(self.SEQ_LENGTH, len(group)):
                    history_seq_release = releases_s[j-self.SEQ_LENGTH : j]
                    history_seq_times = times[j-self.SEQ_LENGTH : j]
                    gru_input = []
                    for k in range(self.SEQ_LENGTH):
                        time_delta = (history_seq_times[k] - history_seq_times[k-1]) if k > 0 else 0
                        gru_input.append(np.concatenate(([history_seq_release[k]], features_s, [time_delta])))
                    input_tensor = torch.from_numpy(np.array([gru_input])).float().to(self.device)
                    gru_hidden_state, _ = gru_extractor.module_.gru(input_tensor)
                    sequence_feature = gru_hidden_state.cpu().numpy()[:, -1, :].flatten()
                    static_features_original = group[self.static_feature_columns].iloc[j].values
                    time_feature = times[j]
                    target = releases_s[j]
                    lgbm_row = np.concatenate([static_features_original, [time_feature], sequence_feature, [target], [group['drug_name'].iloc[0]], [sid]])
                    all_lgbm_samples.append(lgbm_row)
        if not all_lgbm_samples: return pd.DataFrame()
        gru_feature_names = [f'gru_{i}' for i in range(sequence_feature.shape[0])]
        df_lgbm = pd.DataFrame(all_lgbm_samples, columns=self.static_feature_columns + ['time'] + gru_feature_names + ['target', 'drug_name', 'sample_id'])
        for col in self.static_feature_columns + ['time', 'target'] + gru_feature_names:
            df_lgbm[col] = pd.to_numeric(df_lgbm[col])
        return df_lgbm

    def cross_validation(self, num_trials=10, num_prefix_points=3):
        all_sample_ids = self.df['sample_id'].unique()
        drug_groups = self.df.drop_duplicates('sample_id').set_index('sample_id').loc[all_sample_ids]['drug_name'].values
        outer_cv = GroupShuffleSplit(n_splits=num_trials, test_size=0.2, random_state=42)
        for i, (train_ids_idx, test_ids_idx) in enumerate(outer_cv.split(all_sample_ids, groups=drug_groups)):
            train_sample_ids = all_sample_ids[train_ids_idx]
            test_sample_ids = all_sample_ids[test_ids_idx]
            print(f"\n--- 外层交叉验证 {i+1}/{num_trials} ---")
            feature_scaler = MinMaxScaler(); release_scaler = MinMaxScaler()
            df_train = self.df[self.df['sample_id'].isin(train_sample_ids)]
            feature_scaler.fit(df_train[self.static_feature_columns])
            release_scaler.fit(df_train[['release_percentage']])
            X_gru_train, y_gru_train = self._prepare_gru_teacher_forcing_data(train_sample_ids, feature_scaler, release_scaler)
            input_size = X_gru_train.shape[2]
            gru_feature_extractor = NeuralNetRegressor(
                GRUFeatureExtractor, module__input_size=input_size, criterion=torch.nn.MSELoss,
                optimizer=torch.optim.Adam, max_epochs=50, lr=0.01, device=self.device, verbose=0
            )
            gru_feature_extractor.fit(X_gru_train, y_gru_train)
            df_lgbm_train = self._create_lgbm_training_set(gru_feature_extractor, train_sample_ids, feature_scaler, release_scaler)
            if df_lgbm_train.empty: continue
            X_lgbm_train = df_lgbm_train.drop(columns=['target', 'drug_name', 'sample_id'])
            y_lgbm_train = df_lgbm_train['target']
            groups_lgbm_train = df_lgbm_train['drug_name']
            self.lgbm_feature_names = X_lgbm_train.columns.tolist()
            inner_cv = GroupKFold(n_splits=5)
            lgbm_search = RSCV(self.lgbm, self.lgbm_p_grid, n_iter=10, scoring='neg_mean_absolute_error',
                               cv=inner_cv, n_jobs=-1, refit=True, random_state=i)
            lgbm_search.fit(X_lgbm_train, y_lgbm_train, groups=groups_lgbm_train)
            best_lgbm_model = lgbm_search.best_estimator_

            # 准备收集该轮的详细结果
            current_G_test, current_E_test, current_T_test, current_y_test, current_pred = [], [], [], [], []
            all_test_maes = []

            for sid in test_sample_ids:
                true_curve = self.df[self.df['sample_id'] == sid]['release_percentage'].values
                predicted_curve = self._predict_autoregressive_hybrid(
                    best_lgbm_model, gru_feature_extractor, sid, feature_scaler, release_scaler, num_prefix_points
                )
                if predicted_curve is None: continue
                all_test_maes.append(mean_absolute_error(true_curve, predicted_curve))

                # 收集详细信息
                current_G_test.append(self.df[self.df['sample_id'] == sid]['drug_name'].iloc[0])
                current_E_test.append(sid)
                current_T_test.append(self.df[self.df['sample_id'] == sid]['time'].values)
                current_y_test.append(true_curve)
                current_pred.append(predicted_curve)

            avg_mae = np.mean(all_test_maes) if all_test_maes else float('nan')
            print(f"  > 第 {i+1} 轮完成 - 自回归测试集 MAE: {avg_mae:.6f}")

            # 将结果存入类属性列表
            self.itr_number.append(i + 1)
            self.outer_results.append(avg_mae)
            self.inner_results.append(abs(lgbm_search.best_score_))
            self.model_params.append(lgbm_search.best_params_)
            self.G_test_list.append(current_G_test)
            self.E_test_list.append(current_E_test)
            self.T_test_list.append(current_T_test)
            self.y_test_list.append(current_y_test)
            self.pred_list.append(current_pred)

    def _predict_autoregressive_hybrid(self, lgbm_model, gru_extractor, sample_id, feature_scaler, release_scaler, num_prefix_points):
        sample_data = self.df[self.df['sample_id'] == sample_id]
        if len(sample_data) <= num_prefix_points: return None
        static_features = sample_data[self.static_feature_columns].iloc[0].values
        static_features_s = feature_scaler.transform(sample_data[self.static_feature_columns].iloc[[0]])[0]
        times = sample_data['time'].values
        releases_s = release_scaler.transform(sample_data[['release_percentage']]).flatten()
        predicted_releases_s = list(releases_s[:num_prefix_points])
        gru_extractor.module_.eval()
        with torch.no_grad():
            for i in range(num_prefix_points, len(times)):
                history_seq_release = predicted_releases_s[i-self.SEQ_LENGTH : i]
                history_seq_times = times[i-self.SEQ_LENGTH : i]
                gru_input = []
                for k in range(self.SEQ_LENGTH):
                    time_delta = (history_seq_times[k] - history_seq_times[k-1]) if k > 0 else 0
                    gru_input.append(np.concatenate(([history_seq_release[k]], static_features_s, [time_delta])))
                input_tensor = torch.from_numpy(np.array([gru_input])).float().to(self.device)
                gru_hidden_state, _ = gru_extractor.module_.gru(input_tensor)
                sequence_feature = gru_hidden_state.cpu().numpy()[:, -1, :].flatten()
                time_feature = times[i]
                lgbm_input_data = np.concatenate([static_features, [time_feature], sequence_feature]).reshape(1, -1)
                lgbm_input_df = pd.DataFrame(lgbm_input_data, columns=self.lgbm_feature_names)
                next_pred_s = lgbm_model.predict(lgbm_input_df)[0]
                predicted_releases_s.append(next_pred_s)
        return release_scaler.inverse_transform(np.array(predicted_releases_s).reshape(-1, 1)).flatten()

    def results(self):
        print("\n--- 生成交叉验证结果报告 ---")
        list_of_tuples = list(
            zip(self.itr_number, self.inner_results, self.outer_results, self.model_params, self.G_test_list,
                self.E_test_list, self.T_test_list, self.y_test_list, self.pred_list))
        self.CV_dataset = pd.DataFrame(list_of_tuples,
                                       columns=['Iter', 'Valid Score', 'Test Score', 'Model Parms', 'DP_Groups',
                                                "Experimental Index", "Time", 'Experimental_Release', 'Predicted_Release'])
        self.CV_dataset['Score_difference'] = abs(self.CV_dataset['Valid Score'] - self.CV_dataset['Test Score'])
        self.CV_dataset.sort_values(by=['Score_difference', 'Test Score'], ascending=True, inplace=True)
        self.CV_dataset = self.CV_dataset.reset_index(drop=True)
        print(self.CV_dataset.drop(columns=['DP_Groups', 'Experimental Index', 'Time', 'Experimental_Release', 'Predicted_Release']).head())

    def best_model(self):
        print("\n--- 正在训练最终生产模型 ---")
        # 1. 训练最终的GRU
        all_sample_ids = self.df['sample_id'].unique()
        feature_scaler = MinMaxScaler(); release_scaler = MinMaxScaler()
        feature_scaler.fit(self.df[self.static_feature_columns])
        release_scaler.fit(self.df[['release_percentage']])
        X_gru_all, y_gru_all = self._prepare_gru_teacher_forcing_data(all_sample_ids, feature_scaler, release_scaler)
        input_size = X_gru_all.shape[2]
        final_gru_extractor = NeuralNetRegressor(
            GRUFeatureExtractor, module__input_size=input_size, criterion=torch.nn.MSELoss,
            optimizer=torch.optim.Adam, max_epochs=50, lr=0.01, device=self.device, verbose=0
        )
        final_gru_extractor.fit(X_gru_all, y_gru_all)
        # 2. 准备最终的LGBM训练集
        df_all_augmented = self._create_lgbm_training_set(final_gru_extractor, all_sample_ids, feature_scaler, release_scaler)
        # 3. 训练最终的LGBM
        best_params = self.CV_dataset.sort_values(by='Test Score').iloc[0]['Model Parms']
        print(f"  > 使用最佳LGBM参数: {best_params}")
        final_lgbm_model = lgb.LGBMRegressor(random_state=42, verbose=-1, **best_params)
        X_lgbm_all = df_all_augmented.drop(columns=['target', 'drug_name', 'sample_id'])
        y_lgbm_all = df_all_augmented['target']
        final_lgbm_model.fit(X_lgbm_all, y_lgbm_all)

        # 将所有最终组件保存到类属性中
        self.final_model = {
            'gru_extractor': final_gru_extractor,
            'lgbm_predictor': final_lgbm_model,
            'feature_scaler': feature_scaler,
            'release_scaler': release_scaler,
            'lgbm_feature_names': X_lgbm_all.columns.tolist()
        }
        print("最终模型训练完成！")

# --- 运行与保存的总控制函数 ---
def run_hybrid_cv(config):
    model_instance = HybridAutoregressiveCV(
        datafile=config['input_file'],
        static_feature_columns=config['static_features'],
        seq_len=config['seq_len']
    )
    # 运行交叉验证
    model_instance.cross_validation(
        num_trials=config['num_trials'],
        num_prefix_points=config['num_prefix_points']
    )
    # 生成结果报告
    model_instance.results()

    # 保存交叉验证结果
    os.makedirs(config['results_folder'], exist_ok=True)
    result_filename = os.path.join(config['results_folder'], config['results_filename'])
    model_instance.CV_dataset.to_pickle(result_filename)
    print(f"\n交叉验证结果已成功保存至: {result_filename}")

    # 训练并保存最终模型
    model_instance.best_model() # 此方法现在负责训练

    # 保存 final_model 字典
    output_dir = os.path.join(config['models_folder'], config['model_name'])
    os.makedirs(output_dir, exist_ok=True)
    model_filename = os.path.join(output_dir, f"{config['model_name']}_final_model.pkl")
    with open(model_filename, 'wb') as f:
        pickle.dump(model_instance.final_model, f)
    print(f"最终模型管道已成功保存至: {model_filename}")

    return model_instance.CV_dataset, model_instance.final_model

# --- 主程序 ---
if __name__ == '__main__':
    config = {
        "input_file": "release_data/sup_4_release_data.xlsx",
        "static_features": ['Polymer_MW', 'LA/GA', 'Polymer_size', 'Zeta_Potential', 'PDI', 'EE', 'Drug_MW', 'LogP', 'HBD', 'HBA', 'HA', 'RB', 'PSA', 'NAR', 'FSP3'],
        "results_folder": "release_NESTED_CV_RESULTS_hyb",
        "models_folder": "release_Trained_models_hyb",
        "model_name": "Hybrid_Autoregressive_Model",
        "num_trials": 2,
        "seq_len": 3,
        "num_prefix_points": 3
    }
    config["results_filename"] = f"cv_results_{config['model_name']}.pkl"
    if config["num_prefix_points"] < config["seq_len"]:
        raise ValueError("错误: num_prefix_points 不能小于 seq_len")

    set_seed(42)
    result_df, final_model_dict = run_hybrid_cv(config)
    print("\n--- 完整流程执行完毕 ---")