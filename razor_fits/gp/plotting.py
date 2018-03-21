import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch

import razor_data

### UTILITIES

def compute_nsigma(y, samples):
    """
    Computes the p-values of the observed y-value(s)
    with respect to the simulated samples and converts
    it into a number of sigma.
    """
    samples = np.asarray(samples).T
    num_samples = samples.shape[0]
    p = (samples < y).sum(0) / num_samples
    return stats.norm.ppf(p)

def gauss(x, *p):
    """
    Gaussian function with parameters p = (mu, sigma).
    """
    mu, sigma = p
    return ((2 * np.pi * sigma**2)**(-0.5) 
            * np.exp(-(x - mu)**2 / (2 * sigma**2)))

def gauss_fit(bin_centers, counts):
    """
    Performs a Gaussian fit to histogram counts.
    Returns the mean and variance of the fitted Gaussian.
    Adapted from:
    https://stackoverflow.com/questions/11507028/fit-a-gaussian-function
    """
    guess = [0., 1.]
    try:
        (mu, sigma), _ = optimize.curve_fit(gauss, bin_centers, counts, p0=guess)
    except RuntimeError:
        print("Gaussian fit failed for nsigma distribution!")
        mu, sigma = None, None
    return mu, sigma


### PLOTTING FUNCTIONS

def plot_hist_1d(binned, log=True, G=None, title=None,
        num_samples=4000, use_noise=False):
    """
    Input: binned data loaded by Razor1DDataset class.
    Optionally provide a Gaussian Process object, G, with a 
    sample() method, which will produce a prediction
    to overlay on the data.
    """
    centers = binned['u'].numpy()
    counts = binned['y'].numpy()

    fig, ax = plt.subplots(figsize=(8, 5))

    if G is not None:
        quantiles = [2.5, 16, 50, 84, 97.5]
        samples = [G.sample(x, num_samples=num_samples,
            use_noise=use_noise) for x in binned['u']]
        bands = {q:[np.percentile(s, q) for s in samples] for q in quantiles}
        plt.fill_between(centers, bands[2.5], bands[97.5], 
                facecolor='b', alpha=0.35)
        plt.fill_between(centers, bands[16], bands[84], 
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

def plot_nsigma_1d(binned, G, num_samples=40000,
        use_poisson_noise=False):
    """
    Inputs:
        binned: dataset loaded by Razor1DDataset or Razor2DDataset
        G: Gaussian Process object with a sample() method.
        num_samples: number of samples to draw for each nsigma computation
        use_poisson_noise: only set to True if approximating Poisson noise
            using the observed data counts (for Gaussian GP likelihood)
    Plots the distribution of significances between data and fit.
    """
    centers = binned['u']
    counts = binned['y'].numpy()
    if use_poisson_noise:
        samples = [G.sample(x, num_samples=num_samples,
            use_noise=True, poisson_noise_vector=torch.Tensor([float(counts[i])])) 
            for i, x in enumerate(centers)]
    else:
        samples = [G.sample(x, num_samples=num_samples,
            use_noise=True) for x in centers]
    nsigma = compute_nsigma(counts, samples)
    if np.isinf(nsigma).any():
        print("Error: at least one nsigma is infinite")
        return

    bin_width = 0.2
    bin_edges = np.append(np.arange(-4, 4, bin_width), [4.0])
    hist, bin_edges = np.histogram(nsigma, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    fig, ax = plt.subplots(figsize=(8, 6))

    bin_width_fine = 0.01
    bin_edges_fine = np.arange(-4, 4, bin_width_fine)
    mu, sigma = gauss_fit(bin_centers, hist) 
    if mu is not None and sigma is not None:
        fit = gauss(bin_edges_fine, mu, sigma)
        plt.plot(bin_edges_fine, fit, 'r', linewidth=3.0)
        plt.text(0.05, 0.75, "Fitted mean: {:.2f}".format(mu), fontsize=14,
                transform=ax.transAxes)
        plt.text(0.05, 0.80, "Fitted sigma: {:.2f}".format(sigma), fontsize=14,
                transform=ax.transAxes)
    plt.bar(bin_edges[:-1], hist, color='b', align='edge', width=bin_width)
    plt.xlim(-5, 5)
    plt.xlabel('Significance', fontsize=16)
    plt.ylabel('A.U.', fontsize=16)
    ax.tick_params(labelsize=14)
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

def plot_covariance(G, cmap='afmhot', use_noise=False):
    """
    Input: Gaussian Process object
    """
    if use_noise:
        cov = G.Sigma
    else:
        cov = G.K
    cov = cov.data.numpy()
    plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(cov, cmap=cmap, fmt='g',
            xticklabels=G.U.data.numpy().round(-1).astype(int),
            yticklabels=G.U.data.numpy().round(-1).astype(int))

