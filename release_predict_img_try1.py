import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams

# --- 全局绘图参数设置 ---
sz=15

rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = sz+3       # 设置标题字体大小
rcParams['axes.titleweight'] = 'bold' # 设置标题加粗
rcParams['axes.labelsize'] = sz       # 设置x和y轴标签字体大小
rcParams['axes.labelweight'] = 'bold' # 设置x和y轴标签加粗
rcParams['xtick.labelsize'] = sz-3
rcParams['ytick.labelsize'] = sz-3
rcParams['legend.fontsize'] = sz-2

def data_extraction(df, n):
    '''
    Function to extract data from results of the NESTED_CV experiments,
    then return the results for the indicated nested loop - these are ranked in order of score_differeent,
    where score difference is: inner_loop MAE (i.e., valid score) - outer loop MAE (i.e., test score)

    Takes arguments: (i) model_name and (ii) sorted nested loop number (i.e., "n")
    '''

    dataframe = pd.DataFrame(df['Time'][n], columns=['Time'])
    dataframe['Experimental_Release'] = df['Experimental_Release'][n]
    dataframe['Predicted_Release'] = df['Predicted_Release'][n]
    # dataframe['Variance'] = dataframe['Variance'][n]
    dataframe['Experimental Index'] = df['Experimental Index'][n]
    dataframe['DP_Groups'] = df['DP_Groups'][n]

    return dataframe


if __name__ == '__main__':
    MODEL = pd.read_pickle("release_best_cv_results/cv_results_Hybrid_Autoregressive_Model_1_shot.pkl", compression='infer',
                           storage_options=None)

    MODEL_TEST = data_extraction(MODEL, 0)

    # group dataframe by experimental index
    grouped = MODEL_TEST.groupby('Experimental Index')

    # generate predicted versus experimental drug release profiles for the test set
    for name, group in grouped:
        X1 = group['Time']
        Y1 = group['Predicted_Release']
        X2 = group['Time']
        Y2 = group['Experimental_Release']

        dp = group['DP_Groups'].iloc[0]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(X1, Y1, label='Predicted', linestyle='--', marker='o', markersize=8,
                markeredgecolor="black", alpha=0.8)

        ax.plot(X2, Y2, label='Experimental', linestyle='--', marker='o', markersize=8,
                alpha=0.8, markeredgecolor="black")

        # y-axis limits and interval
        ax.set(ylim=(-0.02, 1.08), yticks=np.arange(0, 1.08, 0.1))
        ax.set_xlabel('Time (Hours)', fontsize=15, color='black', weight='bold')
        ax.set_ylabel('Fractional Drug Release', fontsize=15, color='black', weight='bold')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.set_title(str(dp) + ' (Experimental_index' + str(name) + ')', color='black', weight='bold',
                     fontsize=18, pad=20)
        ax.legend(loc='upper left', frameon=False, prop={'size': 12})
        ax.grid(False)

        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)

        # x-axis and y-axis tick color
        ax.tick_params(colors='black')  # 'both' refers to minor and major axes

        plt.tight_layout()
        os.makedirs("release_Figures_real_predict", exist_ok=True)
        plt.savefig('release_Figures_real_predict/cut5_sup_real_hyb_release_EXP-INDEX' + str(name) + '.png', dpi=600,
                    transparent=False)

        plt.close()
