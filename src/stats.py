import os
import sklearn.metrics
import scipy.stats
from glob import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, gls

def main():
    csv_files = [csv_file for csv_file in glob("preds/simple_*.csv")]

    # deperate csv files by frontend again (could have done each file seperately but I prefer this way)
    df = pd.concat((pd.read_csv(f) for f in csv_files))
    frontend_df = [y for x, y in df.groupby('frontend', as_index=False)]
    metrics_dict = {}

    # report accuracies again
    print("---- Metrics ----")
    # for each frontend, calculate acc for each split
    for i in range(len(frontend_df)):
        df = frontend_df[i]
        frontend = df['frontend'].loc[0]
        metrics_list = []
        splits_df = [y for x, y in df.groupby('split', as_index=False)]
        for j in range(len(splits_df)):
            df = splits_df[j]
            y_target = np.asarray(df['target'])
            y_pred = np.asarray(df['pred'])
            acc = sklearn.metrics.accuracy_score(y_target, y_pred)
            metrics_list.append(acc)
        print(f"{frontend} Accuracy: {np.mean(metrics_list)}")
        metrics_dict[frontend] = metrics_list
    input('Press Enter to continue or Ctrl+c to quit...')

    metrics_df = pd.DataFrame(metrics_dict)
    # Shapiro-Wilkes test for normality
    # TODO check with smarter stats people about this
    print("---- Normality Testing ----")
    print('H_0: Samples are Normally Distributed\nH_1: Samples are not Normally Distributed')
    for frontend in metrics_df.columns:
        samples = metrics_df[frontend]
        W, p_shapiro = scipy.stats.shapiro(samples)
        print(f"{frontend}: W={W:.5f}, p-value={p_shapiro:.5f}")
        if p_shapiro < 0.05:
            print(f"WARNING: {frontend} is potentially not normally distributed!")
    input('Press Enter to continue or Ctrl+c to quit...')
    print()

    #additional test data
    #metrics_df['test'] = np.random.normal(0.53, 0.005, size=10)

    # reshape df for R type processing
    metrics_df = pd.melt(metrics_df.reset_index(), id_vars=['index'], value_vars=metrics_df.columns)
    metrics_df.columns=['index', 'frontend', 'value']
    
    # Ordinary Least Squares (OLS) model
    model = ols('value ~ C(frontend)', data=metrics_df).fit()
    #print(model.summary())
    #input('Press Enter to continue or Ctrl+c to quit...')
    #print()

    # ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('---- ANOVA ----')
    print('H_0: All means equal\nH_1: Not all Means Equal')
    print(anova_table)
    input('Press Enter to continue or Ctrl+c to quit...')
    print()
    
    # Tukey HSD test
    tukey_hsd = sm.stats.multicomp.pairwise_tukeyhsd(endog=metrics_df['value'], groups=metrics_df['frontend'])
    print('---- Tukey HSD ----')
    print('H_0: Means equal\nH_1: Means not Equal')
    print(tukey_hsd.summary())

if __name__ == "__main__":
    main()
