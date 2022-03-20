import os
import sklearn.metrics
import scipy.stats
from glob import glob
import numpy as np
import pandas as pd

def main():
    csv_files = [csv_file for csv_file in glob("preds/*.csv")]

    # deperate csv files by frontend again (could have done each file seperately but I prefer this way)
    df = pd.concat((pd.read_csv(f) for f in csv_files))
    frontend_df = [y for x, y in df.groupby('frontend', as_index=False)]
    metrics_dict = {}

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
        metrics_dict[frontend] = metrics_list

    # Shapiro-Wilkes test for normality
    # TODO check with smarter stats people about this
    print("Normality Testing")
    for frontend in metrics_dict:
        samples = metrics_dict[frontend]
        W, p_shapiro = scipy.stats.shapiro(samples)
        print(f"{frontend}: W={W:.5f}, p-value={p_shapiro:.5f}")

    breakpoint()
    # ANOVA (one-way) testing
    F, p_anova = scipy.stats.f_oneway(
        np.random.normal(loc=0.79, scale=0.3, size=(2,1)),
        np.random.normal(loc=0.82, scale=0.25, size=(2,1)),
        np.random.normal(loc=0.82, scale=0.18, size=(2,1))
        #metrics_dict['pcen'],
        #metrics_dict['logmel']
    )

    breakpoint()

if __name__ == "__main__":
    main()
