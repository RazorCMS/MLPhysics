import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import razor_data

def plot_hist_1d(binned, log=True, G=None, title=None,
        num_samples=1000):
    """
    Input: binned data loaded by Razor1DDataset class.
    Optionally provide a Gaussian Process object, G, with a 
    sample() method, which will produce a prediction
    to overlay on the data.
    """
    centers = binned['u'].numpy()
    counts = binned['y'].numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    if G is not None:
        quantiles = [5, 32, 50, 68, 95]
        samples = [G.sample(x, num_samples=num_samples) for x in binned['u']]
        bands = {q:[np.percentile(s, q) for s in samples] for q in quantiles}
        plt.fill_between(centers, bands[5], bands[95], 
                facecolor='b', alpha=0.35)
        plt.fill_between(centers, bands[32], bands[68], 
                facecolor='b', alpha=0.5)
        plt.plot(centers, bands[50], color='b')
    
    ax.plot(centers, counts, 'ko') 
    ax.tick_params(labelsize=14)
    plt.xlabel('MR', fontsize=16)
    plt.ylabel('Counts', fontsize=16)
    if title is not None:
        plt.title(title, fontsize=16)
    if log:
        plt.yscale('log')
    plt.ylim(ymin=0.1)

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
