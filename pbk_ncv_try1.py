import os
os.environ["LOKY_MAX_CPU_COUNT"] = "12"  # 或其他你想用的并行数
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV as RSCV


class NESTED_CV:
    def __init__(self, datafile="PBK_data/pbk_sup_1_data.xlsx"):

        self.df = pd.read_excel(datafile)

        self.user_defined_model = LGBMRegressor(random_state=4, verbose=-1)
        self.p_grid = {"n_estimators": [100, 150, 200, 250, 300, 400, 500, 600],
                       'boosting_type': ['gbdt', 'dart', 'goss'],
                       'num_leaves': [16, 32, 64, 128, 256],
                       'learning_rate': [0.1, 0.01, 0.001, 0.0001],
                       'min_child_weight': [0.001, 0.01, 0.1, 1.0, 10.0],
                       'subsample': [0.4, 0.6, 0.8, 1.0],
                       'min_child_samples': [2, 10, 20, 40, 100],
                       'reg_alpha': [0, 0.005, 0.01, 0.015],
                       'reg_lambda': [0, 0.005, 0.01, 0.015]}

    def input_target(self):
        # X_temp = self.df.drop(['sample_id', 'source', 'drug_name', 'release_percentage'], axis='columns')
        X_temp = self.df.drop(['sample_id', 'source', 'drug_name', 'release_percentage',
                               'EMW', 'LR_HBA', 'LR_HBD', 'LR_LogP', 'LR_MW', 'LA/GA', 'HA'], axis='columns')
        stdScale = StandardScaler()
        stdScale.set_output(transform="pandas")
        self.X = stdScale.fit_transform(X_temp)
        self.Y = self.df['release_percentage']
        # 打乱尝试
        # self.Y = self.Y.sample(frac=1, random_state=42).reset_index(drop=True)

        self.G = self.df['drug_name'].str.strip().str.lower()  # 去除空格并统一小写
        self.E = self.df['sample_id']
        self.T = self.df['time']


    def cross_validation(self, input_value):
        if input_value == None:
            NUM_TRIALS = 10
        else:
            NUM_TRIALS = input_value

        self.itr_number = []  # create new empty list for itr number
        self.outer_results = []
        self.inner_results = []
        self.model_params = []
        self.G_test_list = []
        self.y_test_list = []
        self.E_test_list = []
        self.T_test_list = []
        self.pred_list = []

        for i in range(NUM_TRIALS):  # configure the cross-validation procedure - outer loop (test set)
            print(f"\n开始第 {i + 1}/{NUM_TRIALS} 次外层交叉验证...")
            cv_outer = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                         random_state=i)  # hold back 20% of the groups for test set

            # split data using GSS
            for train_index, test_index in cv_outer.split(self.X, self.Y, self.G):
                # X_train, X_test = self.X[train_index], self.X[test_index]
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.Y[train_index], self.Y[test_index]
                G_train, G_test = self.G[train_index], self.G[test_index]
                E_train, E_test = self.E[train_index], self.E[test_index]
                T_train, T_test = self.T[train_index], self.T[test_index]

                # store test set information
                G_test = np.array(G_test)  # prevents index from being brought from dataframe
                self.G_test_list.append(G_test)
                E_test = np.array(E_test)  # prevents index from being brought from dataframe
                self.E_test_list.append(E_test)
                T_test = np.array(T_test)  # prevents index from being brought from dataframe
                self.T_test_list.append(T_test)
                y_test = np.array(y_test)  # prevents index from being brought from dataframe
                self.y_test_list.append(y_test)

                # configure the cross-validation procedure - inner loop (validation set/HP optimization)
                cv_inner = GroupKFold(n_splits=10)  # should be 10 fold group split for inner loop

                # define search space
                search = RSCV(self.user_defined_model, self.p_grid, n_iter=100, verbose=0,
                              scoring='neg_mean_absolute_error', cv=cv_inner, n_jobs=-1, refit=True, random_state=i)  # should be 100

                # execute search
                result = search.fit(X_train, y_train, groups=G_train)

                # get the best performing model fit on the whole training set
                best_model = result.best_estimator_

                # get the score for the best performing model and store
                best_score = abs(result.best_score_)
                self.inner_results.append(best_score)

                # evaluate model on the hold out dataset
                yhat = best_model.predict(X_test)

                # store drug release predictions
                self.pred_list.append(yhat)

                # evaluate the model
                acc = mean_absolute_error(y_test, yhat)

                # store the result
                self.itr_number.append(i + 1)
                self.outer_results.append(acc)
                self.model_params.append(result.best_params_)

                # report progress at end of each inner loop
                print('\n################################################################\n\nSTATUS REPORT:')
                print('状态报告：')
                print(f'迭代 {i+1}/{NUM_TRIALS} 完成')
                print(f'测试集得分 (MAE): {acc:.5f}, 最佳验证集得分 (MAE): {best_score:.5f}')
                print(f'\n最佳模型参数:\n{result.best_params_}')
                print("\n################################################################\n ")


    def results(self):
        # create dataframe with results of nested CV
        list_of_tuples = list(
            zip(self.itr_number, self.inner_results, self.outer_results, self.model_params, self.G_test_list,
                self.E_test_list, self.T_test_list, self.y_test_list, self.pred_list))
        CV_dataset = pd.DataFrame(list_of_tuples, columns=['Iter', 'Valid Score', 'Test Score', 'Model Parms', 'DP_Groups',
                                                           "Experimental Index", "Time", 'Experimental_Release',
                                                           'Predicted_Release'])
        CV_dataset['Score_difference'] = abs(CV_dataset['Valid Score'] - CV_dataset['Test Score'])  # Groupby dataframe model iterations that best fit the data (i.e., validitaion <= test)
        CV_dataset.sort_values(by=['Score_difference', 'Test Score'], ascending=True, inplace=True)
        CV_dataset = CV_dataset.reset_index(drop=True)  # Reset index of dataframe
        # save the results as a class object
        self.CV_dataset = CV_dataset


    def best_model(self):
        # assign the best model paramaters
        best_model_params = self.CV_dataset.iloc[0, 3]
        # set params from the best model to a class object
        best_model = self.user_defined_model.set_params(**best_model_params)
        self.final_model = best_model.fit(self.X, self.Y)


def run_NESTED_CV(name,CV):
    model_instance = NESTED_CV()
    model_instance.input_target()
    model_instance.cross_validation(CV)
    model_instance.results()
    model_instance.best_model()
    # 创建目录（如果不存在）
    os.makedirs("release_NESTED_CV_RESULTS", exist_ok=True)
    os.makedirs("release_Trained_models", exist_ok=True)
    model_instance.CV_dataset.to_pickle("PBK_NESTED_CV_RESULTS/pbk_cut7_sup_1_data_" + str(name) + ".pkl", compression='infer',
                                        protocol=5, storage_options=None)  # save dataframe as pickle file
    with open('PBK_Trained_models/pbk_cut7_sup_1_data_' + str(name) + '_model.pkl', 'wb') as file:  # Save the Model to pickle file
        pickle.dump(model_instance.final_model, file)
    return model_instance.CV_dataset, model_instance.best_model


if __name__ == '__main__':
    NESTED_CV()
    LGBM_result, LGBM_model = run_NESTED_CV("LGBM", None)
    if LGBM_result is not None and LGBM_model is  not None:
        print("\n嵌套交叉验证流程完成！")
        print("\nCV 结果摘要:")
        print(LGBM_result.head())
        print("\n最佳模型:")
        print(LGBM_model)
    else:
        print("\n嵌套交叉验证流程未能成功完成。")

