import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import razor_data

def plot_hist_1d(binned, log=True, title=None):
    """
    Input: binned data loaded by Razor1DDataset class
    """
    centers = binned['u'].numpy()
    counts = binned['y'].numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(centers, counts, 'ko') 
    ax.tick_params(labelsize=14)
    plt.xlabel('MR', fontsize=16)
    plt.ylabel('Counts', fontsize=16)
    if title is not None:
        plt.title(title, fontsize=16)
    if log:
        plt.yscale('log')
    plt.show()

def plot_hist_2d(binned, annot=False, cmap='afmhot', title=None):
    """
    Input: binned data loaded by Razor2DDataset class
    """
    df = razor_data.binned_data_to_df(binned)
    pivoted = df.pivot(index='rsq_center', columns='mr_center', 
            values='counts')

    plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(pivoted, annot=annot, cmap=cmap, fmt='g',
            xticklabels=pivoted.columns.values.round(-1).astype(int),
            yticklabels=pivoted.index.values.round(2))
    ax.tick_params(labelsize=14)
    ax.invert_yaxis()
    plt.xlabel('MR', fontsize=16)
    plt.ylabel('Rsq', fontsize=16)
    if title is not None:
        plt.title(title, fontsize=16)
    plt.show()
