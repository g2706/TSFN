import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
    LGBM = pd.read_pickle("PBK_NESTED_CV_RESULTS/pbk_sup_1_data_LGBM.pkl", compression='infer', storage_options=None)

    LGBM_test = data_extraction(LGBM, 0)

    # group dataframe by experimental index
    grouped = LGBM_test.groupby('Experimental Index')

    # generate predicted versus experimental drug release profiles for the test set
    for name, group in grouped:
        X1 = group['Time']
        Y1 = group['Predicted_Release']
        X2 = group['Time']
        Y2 = group['Experimental_Release']
        # Y1_err=group['Variance']

        dp = group['DP_Groups'].iloc[1]

        # Compute upper and lower bounds using chosen uncertainty measure: here
        # it is a fraction of the standard deviation of measurements at each
        # time point based on the unbiased sample variance
        # lower = Y1 - Y1_err
        # upper = Y1 + Y1_err

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(X1, Y1, label='Predicted (LGBM)', linestyle='--', marker='o', markersize=8,
                markeredgecolor="black", alpha=0.8)
        # ax.plot(X1, lower, color='tab:blue', alpha=0.2)
        # ax.plot(X1, upper, color='tab:blue', alpha=0.2)
        # ax.fill_between(X1, lower, upper, alpha=0.3, label='Variance')

        ax.plot(X2, Y2, label='Experimental', linestyle='--', marker='o', markersize=8,
                alpha=0.8, markeredgecolor="black")

        # y-axis limits and interval
        # ax.set(ylim=(-0.02, 1.08), yticks=np.arange(0, 1.08, 0.1))
        ax.set_xlabel('Time (Days)', fontsize=15, color='black', weight='bold')
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
        os.makedirs("PBK_Figures_LGBM_predict", exist_ok=True)
        plt.savefig('PBK_Figures_LGBM_predict/sup_1_LGBM_PBK_EXP-INDEX' + str(name) + '.png', dpi=600, transparent=False)

        plt.close()
