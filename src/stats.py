import os
import argparse
import sklearn.metrics
import scipy.stats
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols, gls

def main(args):
    csv_files = [csv_file for csv_file in glob("preds/efficient_*.csv")]

    # deperate csv files by frontend again (could have done each file seperately but I prefer this way)
    df = pd.concat((pd.read_csv(f) for f in csv_files))
    frontend_df = [y for x, y in df.groupby('frontend', as_index=False)]
    metrics_dict = {}

    metric_per_source = np.zeros((len(df['datasetid'].unique())+1, len(frontend_df)))

    # report accuracies again
    print("---- Metrics ----")
    # for each frontend, calculate acc for each split
    for i in range(len(frontend_df)):
        df = frontend_df[i]
        frontend = df['frontend'].loc[0]
        metrics_list = []

        splits_df = [y for x, y in df.groupby('split', as_index=False)]
        dataset_df = [y for x, y in df.groupby('datasetid', as_index=False)]
        #breakpoint()
        for j in range(len(splits_df)):
            df = splits_df[j]
            y_target = np.asarray(df['target'])
            y_pred = np.asarray(df['pred'])
            acc = sklearn.metrics.accuracy_score(y_target, y_pred)
            metrics_list.append(acc)
        print(f"{frontend} Accuracy: {np.mean(metrics_list)}, StdDev: {np.std(metrics_list)}")
        metrics_dict[frontend] = metrics_list
        metric_per_source[3, i] = np.mean(metrics_list)

        for k in range(len(dataset_df)):
            df = dataset_df[k]
            y_target = np.asarray(df['target'])
            y_pred = np.asarray(df['pred'])
            acc = sklearn.metrics.accuracy_score(y_target, y_pred)
            metric_per_source[k, i] = acc

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
    print()

    # Tukey HSD test
    tukey_hsd = sm.stats.multicomp.pairwise_tukeyhsd(endog=metrics_df['value'], groups=metrics_df['frontend'])
    print('---- Tukey HSD ----')
    print('H_0: Means equal\nH_1: Means not Equal')
    print(tukey_hsd.summary())
    df_pvalues = pd.DataFrame(
        data=tukey_hsd._results_table.data[1:],
        columns=tukey_hsd._results_table.data[0]
    )
    #generate_table(df_pvalues)

    p_tukey = tukey_hsd.pvalues
    c_tukey_95 = p_tukey#np.where(p_tukey <= 0.05, 1. - p_tukey, 0)
    groups = np.unique(tukey_hsd.groups)
    c_tukey_95 = np.split(c_tukey_95, [6,11,15,18,20,21])
    tukey_matrix = np.empty((len(groups), len(groups))) * np.nan
    for i in range(len(groups)):
        tukey_matrix[i+1:,i] = c_tukey_95[i]

    if not args.no_graph:
        permutation = [4, 2, 1, 5, 6, 3, 0]
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))
        metric_per_source[:] = metric_per_source[:, idx]
        metric_per_source = metric_per_source[(1,2,0,3),:]

        frontends = ['spect', 'mel', 'logmel', 'STRF', 'TD', 'PCEN', 'LEAF']
        sources = ['freefield1010', 'warblrb10k', 'BirdVox-DCASE-20k', 'Total Acc']

        #fig, ax1 = plt.subplots(1, 1)
        #ax1.set_title('Pairwise-Comparisons of Frontends (p-value)')
        #tukey_heatmap = sns.heatmap(
        #    tukey_matrix,
        #    annot=np.abs(tukey_matrix),
        #    xticklabels=groups,
        #    yticklabels=groups,
        #    vmin=0, vmax=1,
        #    ax=ax1
        #)
        #metric_per_source = metric_per_source[:-1]
        metric_per_source = np.around(metric_per_source*100, decimals=1)
        #sources = sources[:-1]
        x = np.arange(len(frontends))  # the label locations
        width = 0.18  # the width of the bars
        fig, ax = plt.subplots(figsize=(16, 10))
        rects1 = ax.bar(x - 1.5*width, metric_per_source[3], width, label=sources[3], facecolor='#a6cee3', hatch='/')
        rects2 = ax.bar(x - 0.5*width, metric_per_source[0], width, label=sources[0], facecolor='#1f78b4', hatch='\\')
        rects3 = ax.bar(x + 0.5*width, metric_per_source[1], width, label=sources[1], facecolor='#b2df8a', hatch='x')
        rects4 = ax.bar(x + 1.5*width, metric_per_source[2], width, label=sources[2], facecolor='#33a02c', hatch='.')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracy %', fontsize=24)
        ax.set_title('Accuracy by frontend and dataset', fontsize=26)
        ax.set_xticks(x, frontends, fontsize=22)
        ax.tick_params(axis="y", labelsize=20)
        ax.legend(fontsize=16, loc=2, prop={'size': 26})
        ax.set_ylim(40, 100)
        ax.grid(True, which='major', axis='y')
        # ax.bar_label(rects1, fontweight='bold', fontsize=10, padding=1)
        # ax.bar_label(rects2, fontsize=10, padding=1)
        # ax.bar_label(rects3, fontsize=10, padding=1)
        # ax.bar_label(rects4, fontsize=10, padding=1)
        fig.tight_layout()
        plt.savefig('figures/results_frontend_dataset.png', dpi=600)
        plt.show()

def generate_table(df_pvalues):
    systems = df_pvalues['group2'].unique()
    n_systems = len(systems)
    with open('tukey_table.tex', 'w') as f_out:
        print("\\begin{tabular}{%s}" % ("c" * (n_systems + 1)), file=f_out)
        print(" & ".join([""] + systems) + " \\\\", file=f_out)
        for i, sys_ref in enumerate(systems):
            print(sys_ref, end=" & ", file=f_out)
            for j, sys_oth in enumerate(systems):
                end = " & "
                if j == (n_systems - 1):
                    end = ""
                # Check border cases
                if i == j:
                    print("NA", end=end, file=f_out)
                    continue
                elif i < j:
                    print("", end=end, file=f_out)
                    continue

                tukey_cond = (df_pvalues['group2'] == sys_ref) & (df_pvalues['group1'] == sys_oth)
                tukey_p_value = df_pvalues[tukey_cond]['p-adj'].iloc[0]

                if tukey_p_value < 0.05:
                    print('$\\blacksquare$', end=end, file=f_out)
                else:
                    print('', end=end, file=f_out)

            print('\\\\', file=f_out)

        print('\\end{tabular}', file=f_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-graph', action='store_true')
    args = parser.parse_args() 
    #if not args.no_graph:
        #matplotlib.use('QT5Agg',force=True)
    main(args)
