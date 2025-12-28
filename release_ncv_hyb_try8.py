import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV as RSCV, GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
from skorch import NeuralNetRegressor
import joblib
# 指定测试集
class IdentityScaler:
    def fit(self, data): return self
    def transform(self, data): return data.values if isinstance(data, pd.DataFrame) else data
    def inverse_transform(self, data): return data

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为: {seed}")

class GRUFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(GRUFeatureExtractor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.to(torch.float32)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class HybridAutoregressiveCV:
    def __init__(self, datafile, static_feature_columns, seq_len, gru_p_grid, lgbm_p_grid):
        self.df = pd.read_excel(datafile)
        self.df['drug_name'] = self.df['drug_name'].str.strip().str.lower()
        self.SEQ_LENGTH = seq_len
        self.static_feature_columns = static_feature_columns
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gru_p_grid = gru_p_grid
        self.lgbm_p_grid = lgbm_p_grid
        self.lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1)
        self.lgbm_feature_names = None
        self.gru_params_list = [] # <--- 新增
        self.itr_number, self.outer_results, self.inner_results, self.model_params = [], [], [], []
        self.G_test_list, self.y_test_list, self.pred_list, self.E_test_list, self.T_test_list = [], [], [], [], []

    # <--- 修改: 增加 time_delta_scaler 作为参数
    def _prepare_gru_teacher_forcing_data(self, sample_ids, feature_scaler, release_scaler, time_delta_scaler):
        X_list, y_list, groups_list = [], [], []
        for sid in sample_ids:
            group = self.df[self.df['sample_id'] == sid]
            drug_name = group['drug_name'].iloc[0]
            features_s = feature_scaler.transform(group[self.static_feature_columns])[0]
            releases_s = release_scaler.transform(group[['release_percentage']]).flatten()
            times = group['time'].values
            if len(releases_s) > self.SEQ_LENGTH:
                for j in range(len(releases_s) - self.SEQ_LENGTH):
                    seq_in_release = releases_s[j:j+self.SEQ_LENGTH]
                    combined_seq = []
                    for k in range(self.SEQ_LENGTH):
                        time_delta = times[j+k+1] - times[j+k]
                        # <--- 修改: 对 time_delta 进行标准化
                        time_delta_s = time_delta_scaler.transform([[time_delta]])[0, 0]
                        combined_seq.append(np.concatenate(([seq_in_release[k]], features_s, [time_delta_s])))
                    X_list.append(np.array(combined_seq))
                    y_list.append(releases_s[j+self.SEQ_LENGTH])
                    groups_list.append(drug_name)
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1), np.array(groups_list)

    # <--- 修改: 增加 time_delta_scaler 作为参数
    def _create_lgbm_training_set(self, gru_extractor, sample_ids, feature_scaler, release_scaler, time_delta_scaler):
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
                        # <--- 修改: 对 time_delta 进行标准化
                        time_delta_s = time_delta_scaler.transform([[time_delta]])[0, 0]
                        gru_input.append(np.concatenate(([history_seq_release[k]], features_s, [time_delta_s])))
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

    # 在 HybridAutoregressiveCV 类中

    def cross_validation(self, num_trials=10, num_prefix_points=3, manual_test_ids=None):
        all_sample_ids = self.df['sample_id'].unique()

        # <--- 修改: 引入一个列表来存储待处理的数据划分方案
        splits_to_process = []

        if manual_test_ids and len(manual_test_ids) > 0:
            print("\n--- 模式: 手动指定测试集 ---")

            all_ids_set = set(all_sample_ids)
            manual_ids_set = set(manual_test_ids)

            missing_ids = manual_ids_set - all_ids_set
            if missing_ids:
                raise ValueError(f"错误：您在'manual_test_sample_ids'中指定的以下测试ID在数据文件中不存在: {sorted(list(missing_ids))}")

            test_sample_ids = np.array(manual_test_ids)
            train_val_sample_ids = np.setdiff1d(all_sample_ids, test_sample_ids)

            print(f"  > 手动指定测试样本数: {len(test_sample_ids)}")
            print(f"  > 剩余训练/验证样本数: {len(train_val_sample_ids)}")

            all_sample_ids_list = list(all_sample_ids)
            train_ids_idx = [all_sample_ids_list.index(sid) for sid in train_val_sample_ids]
            test_ids_idx = [all_sample_ids_list.index(sid) for sid in test_sample_ids]

            # 将这唯一的一次划分方案添加到列表中
            splits_to_process.append((train_ids_idx, test_ids_idx))
            num_trials = 1 # 强制后续的打印信息只显示为1轮
        else:
            print("\n--- 模式: 随机分组交叉验证 ---")
            drug_groups = self.df.drop_duplicates('sample_id').set_index('sample_id').loc[all_sample_ids]['drug_name'].values
            outer_cv = GroupShuffleSplit(n_splits=num_trials, test_size=0.2, random_state=42)
            # 将所有随机划分的方案添加到列表中
            splits_to_process = list(outer_cv.split(all_sample_ids, groups=drug_groups))

        # <--- 修改: 循环现在统一处理 splits_to_process 列表，无论它是如何生成的
        for i, (train_ids_idx, test_ids_idx) in enumerate(splits_to_process):
            #
            # !!! 循环内部的所有代码都保持原样，无需任何改动 !!!
            #
            train_sample_ids = all_sample_ids[train_ids_idx]
            test_sample_ids = all_sample_ids[test_ids_idx]
            # <--- 修改: 打印标题的 num_trials 会根据模式自动调整
            print(f"\n--- 第 {i+1}/{num_trials} 轮 ---")
            feature_scaler = StandardScaler(); release_scaler = IdentityScaler()
            time_delta_scaler = StandardScaler()
            df_train = self.df[self.df['sample_id'].isin(train_sample_ids)]

            time_deltas_train = df_train.groupby('sample_id')['time'].diff().dropna()
            time_delta_scaler.fit(time_deltas_train.values.reshape(-1, 1))

            feature_scaler.fit(df_train[self.static_feature_columns])
            release_scaler.fit(df_train[['release_percentage']])

            X_gru_train, y_gru_train, G_gru_train = self._prepare_gru_teacher_forcing_data(train_sample_ids, feature_scaler, release_scaler, time_delta_scaler)
            input_size = X_gru_train.shape[2]

            print("  > 正在为GRU特征提取器搜索最佳参数...")
            base_gru_net = NeuralNetRegressor(GRUFeatureExtractor, module__input_size=input_size, criterion=torch.nn.L1Loss, optimizer=torch.optim.Adam, device=self.device, verbose=0)
            gru_search = RSCV(base_gru_net, self.gru_p_grid, n_iter=10, scoring='neg_mean_absolute_error', cv=GroupKFold(n_splits=10), n_jobs=-1, refit=True, random_state=i)
            gru_search.fit(X_gru_train, y_gru_train, groups=G_gru_train)
            gru_feature_extractor = gru_search.best_estimator_
            print(f"  > 最佳GRU参数: {gru_search.best_params_}")

            df_lgbm_train = self._create_lgbm_training_set(gru_feature_extractor, train_sample_ids, feature_scaler, release_scaler, time_delta_scaler)
            if df_lgbm_train.empty: continue
            X_lgbm_train = df_lgbm_train.drop(columns=['target', 'drug_name', 'sample_id'])
            y_lgbm_train = df_lgbm_train['target']
            groups_lgbm_train = df_lgbm_train['drug_name']
            self.lgbm_feature_names = X_lgbm_train.columns.tolist()
            inner_cv = GroupKFold(n_splits=10)
            lgbm_search = RSCV(self.lgbm, self.lgbm_p_grid, n_iter=30, scoring='neg_mean_absolute_error', cv=inner_cv, n_jobs=-1, refit=True, random_state=i)
            lgbm_search.fit(X_lgbm_train, y_lgbm_train, groups=groups_lgbm_train)
            best_lgbm_model = lgbm_search.best_estimator_
            nested_G, nested_E, nested_T, nested_y, nested_pred = [], [], [], [], []
            all_test_maes = []
            for sid in test_sample_ids:
                true_curve = self.df[self.df['sample_id'] == sid]['release_percentage'].values
                predicted_curve = self._predict_autoregressive_hybrid(best_lgbm_model, gru_feature_extractor, sid, feature_scaler, release_scaler, time_delta_scaler, num_prefix_points)
                if predicted_curve is None: continue
                all_test_maes.append(mean_absolute_error(true_curve, predicted_curve))
                nested_G.append(self.df[self.df['sample_id'] == sid]['drug_name'].iloc[0])
                nested_E.append(sid)
                nested_T.append(self.df[self.df['sample_id'] == sid]['time'].values)
                nested_y.append(true_curve)
                nested_pred.append(predicted_curve)
            avg_mae = np.mean(all_test_maes) if all_test_maes else float('nan')
            print(f"  > 第 {i+1} 轮完成 - 自回归测试集 MAE: {avg_mae:.6f}")
            flat_G, flat_E, flat_T, flat_y, flat_pred = [], [], [], [], []
            for j in range(len(nested_E)):
                num_points = len(nested_T[j])
                flat_G.extend([nested_G[j]] * num_points); flat_E.extend([nested_E[j]] * num_points); flat_T.extend(nested_T[j]); flat_y.extend(nested_y[j]); flat_pred.extend(nested_pred[j])
            self.itr_number.append(i + 1); self.outer_results.append(avg_mae); self.inner_results.append(abs(lgbm_search.best_score_)); self.model_params.append(lgbm_search.best_params_)
            self.gru_params_list.append(gru_search.best_params_)
            self.G_test_list.append(np.array(flat_G)); self.E_test_list.append(np.array(flat_E)); self.T_test_list.append(np.array(flat_T)); self.y_test_list.append(np.array(flat_y)); self.pred_list.append(np.array(flat_pred))
            if manual_test_ids and len(manual_test_ids) > 0:
                print("  > [手动模式] 已将本轮训练的模型捕获为最终模型。")
            self.final_model = {
                'gru_extractor': gru_feature_extractor, # 这是本轮最优的GRU
                'lgbm_predictor': best_lgbm_model,     # 这是本轮最优的LGBM
                'feature_scaler': feature_scaler,
                'release_scaler': release_scaler,
                'time_delta_scaler': time_delta_scaler,
                'lgbm_feature_names': self.lgbm_feature_names
            }
    # <--- 修改: 增加 time_delta_scaler 作为参数
    def _predict_autoregressive_hybrid(self, lgbm_model, gru_extractor, sample_id, feature_scaler, release_scaler, time_delta_scaler, num_prefix_points):
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
                    # <--- 修改: 对 time_delta 进行标准化
                    time_delta_s = time_delta_scaler.transform([[time_delta]])[0, 0]
                    gru_input.append(np.concatenate(([history_seq_release[k]], static_features_s, [time_delta_s])))
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
        list_of_tuples = list(zip(self.itr_number, self.inner_results, self.outer_results, self.gru_params_list,self.model_params, self.G_test_list, self.E_test_list, self.T_test_list, self.y_test_list, self.pred_list))
        self.CV_dataset = pd.DataFrame(list_of_tuples, columns=['Iter', 'Valid Score', 'Test Score', 'GRU Parms', 'LGBM Parms', 'DP_Groups', "Experimental Index", "Time", 'Experimental_Release', 'Predicted_Release'])
        self.CV_dataset['Score_difference'] = abs(self.CV_dataset['Valid Score'] - self.CV_dataset['Test Score'])
        self.CV_dataset.sort_values(by=['Score_difference', 'Test Score'], ascending=True, inplace=True)
        self.CV_dataset = self.CV_dataset.reset_index(drop=True)
        print("结果报告摘要："); print(self.CV_dataset.drop(columns=['DP_Groups', 'Experimental Index', 'Time', 'Experimental_Release', 'Predicted_Release']).head())

    # 在 best_model 方法中
    def best_model(self):
        print("\n--- 正在训练最终生产模型 ---")

        # <--- 修改: 从CV结果中获取表现最好那一轮的所有最佳参数
        best_fold_params = self.CV_dataset.iloc[0]
        best_gru_params = best_fold_params['GRU Parms']
        best_lgbm_params = best_fold_params['LGBM Parms']
        print(f"  > 使用来自最佳验证折的GRU参数: {best_gru_params}")
        print(f"  > 使用来自最佳验证折的LGBM参数: {best_lgbm_params}")

        all_sample_ids = self.df['sample_id'].unique()
        feature_scaler = StandardScaler(); release_scaler = IdentityScaler()
        time_delta_scaler = StandardScaler()
        time_deltas_all = self.df.groupby('sample_id')['time'].diff().dropna()
        time_delta_scaler.fit(time_deltas_all.values.reshape(-1, 1))

        feature_scaler.fit(self.df[self.static_feature_columns])
        release_scaler.fit(self.df[['release_percentage']])

        X_gru_all, y_gru_all, _ = self._prepare_gru_teacher_forcing_data(all_sample_ids, feature_scaler, release_scaler, time_delta_scaler)
        input_size = X_gru_all.shape[2]

        # <--- 修改: 使用最佳参数来实例化和训练最终的GRU模型
        final_gru_extractor = NeuralNetRegressor(
            GRUFeatureExtractor,
            module__input_size=input_size,
            criterion=torch.nn.L1Loss,
            optimizer=torch.optim.Adam,
            device=self.device,
            verbose=0,
            **best_gru_params  # <--- 使用搜索到的最佳GRU参数
        )
        print("  > 正在训练最终的GRU特征提取器...")
        final_gru_extractor.fit(X_gru_all, y_gru_all)

        # ... 后续的LGBM训练逻辑不变 ...
        df_all_augmented = self._create_lgbm_training_set(final_gru_extractor, all_sample_ids, feature_scaler, release_scaler, time_delta_scaler)

        # <--- 修改: 这里可以直接使用上面获取的best_lgbm_params
        final_lgbm_model = lgb.LGBMRegressor(random_state=42, verbose=-1, **best_lgbm_params)

        X_lgbm_all = df_all_augmented.drop(columns=['target', 'drug_name', 'sample_id'])
        y_lgbm_all = df_all_augmented['target']
        print("  > 正在训练最终的LGBM预测器...")
        final_lgbm_model.fit(X_lgbm_all, y_lgbm_all)

        self.final_model = {
            'gru_extractor': final_gru_extractor,
            'lgbm_predictor': final_lgbm_model,
            'feature_scaler': feature_scaler,
            'release_scaler': release_scaler,
            'time_delta_scaler': time_delta_scaler,
            'lgbm_feature_names': X_lgbm_all.columns.tolist()
        }
        print("最终模型训练完成！")

def run_hybrid_cv(config):
    model_instance = HybridAutoregressiveCV(
        datafile=config['input_file'],
        static_feature_columns=config['static_features'],
        seq_len=config['seq_len'],
        gru_p_grid=config['gru_hyperparameters'],
        lgbm_p_grid=config['lgbm_hyperparameters']
    )
    model_instance.cross_validation(
        num_trials=config['num_trials'],
        num_prefix_points=config['num_prefix_points'],
        manual_test_ids=config.get('manual_test_sample_ids', None)
    )
    model_instance.results()
    os.makedirs(config['results_folder'], exist_ok=True)
    result_filename = os.path.join(config['results_folder'], config['results_filename'])
    model_instance.CV_dataset.to_pickle(result_filename)
    print(f"\n交叉验证结果已成功保存至: {result_filename}")
    model_instance.best_model()
    output_dir = os.path.join(config['models_folder'], config['model_name'])
    os.makedirs(output_dir, exist_ok=True)
    model_filename = os.path.join(output_dir, f"{config['model_name']}_final_model.pkl")
    with open(model_filename, 'wb') as f: pickle.dump(model_instance.final_model, f)
    print(f"最终模型管道已成功保存至: {model_filename}")
    return model_instance.CV_dataset, model_instance.final_model

if __name__ == '__main__':
    config = {
        "input_file": "release_data/sup_6_release_data.xlsx",
        "static_features": ['Polymer_MW', 'LA/GA', 'Polymer_size', 'Zeta_Potential', 'PDI', 'EE', 'Drug_MW', 'LogP', 'HBD', 'HBA', 'HA', 'RB', 'PSA', 'NAR', 'FSP3'],
        "results_folder": "release_NESTED_CV_RESULTS_hyb",
        "models_folder": "release_Trained_models_hyb",
        "model_name": "Hybrid_Autoregressive_Model_3_shot",
        "num_trials": 1, "seq_len": 4, "num_prefix_points": 4,
        "manual_test_sample_ids":[51,52,54,56,59,60,61,68,71,72,93,94],
        "gru_hyperparameters": {
            'max_epochs': [100, 150], 'lr': [0.001, 0.005, 0.01],
            'batch_size': [16, 32], 'module__hidden_size': [16, 32, 48, 64],
            'module__num_layers': [1, 2],
        },
        "lgbm_hyperparameters": {
            'n_estimators': [100, 200, 300, 500], 'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [16, 31, 64], 'max_depth': [-1, 5, 10],
            'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': [0.8, 0.9, 1.0]
            # "n_estimators": [100, 150, 200, 250, 300, 400, 500, 600],
            # 'boosting_type': ['gbdt', 'dart', 'goss'],
            # 'num_leaves': [16, 32, 64, 128, 256],
            # 'learning_rate': [0.1, 0.01, 0.001, 0.0001],
            # 'min_child_weight': [0.001, 0.01, 0.1, 1.0, 10.0],
            # 'subsample': [0.4, 0.6, 0.8, 1.0],
            # 'min_child_samples': [2, 10, 20, 40, 100],
            # 'reg_alpha': [0, 0.005, 0.01, 0.015],
            # 'reg_lambda': [0, 0.005, 0.01, 0.015]
        }
    }
    config["results_filename"] = f"cv_results_{config['model_name']}.pkl"
    if config["num_prefix_points"] < config["seq_len"]:
        raise ValueError("错误: num_prefix_points 不能小于 seq_len")
    set_seed(42)
    result_df, final_model_dict = run_hybrid_cv(config)
    print("\n--- 完整流程执行完毕 ---")