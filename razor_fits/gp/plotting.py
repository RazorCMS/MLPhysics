import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
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
    if not isinstance(samples, np.ndarray):
        # assume it's been created by GaussianProcess.sample
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

def plot_hist_1d(binned=None, U=None, Y=None, S=None, Z=None,
        G=None, num_samples=4000, use_noise=False,
        samples=None, samples_withsignal=None, best_mu=None,
        title=None, verbose=False, log=True, show=True, ymin=0.1):
    """
    Input: binned data loaded by Razor1DDataset class
        OR torch Tensors U and Y
    Optionally provide a Gaussian Process object, G, with a 
    sample() method, which will produce a prediction
    to overlay on the data.
    Alternatively, provide a list of samples directly.
    Optionally provide a signal shape to superimpose on the plot.
    """
    if binned is None: # make our binned dataset out of U and Y
        binned = {'u':U, 'y':Y}
    centers = binned['u'].numpy()
    bin_width = centers[1] - centers[0]
    edges = centers - bin_width / 2
    counts = binned['y'].numpy()

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[1, 0])

    # Variational inducing point positions
    if Z is not None:
        for x in Z.numpy():
            ax0.axvline(x=x, alpha=0.3)

    # Shaded signal shape histogram
    if S is not None:
        signal = S.numpy()
        ax0.bar(edges, signal, width=bin_width, align='edge',
                linewidth=0, label='Signal', facecolor='darkred', alpha=0.35)

    # Best fit with fitted signal prediction included
    if samples_withsignal is not None:
        best_fit = np.percentile(samples_withsignal, 50, axis=0)
        ax0.plot(centers, best_fit, color='forestgreen', alpha=0.8, linewidth=3,
                label='GP + Signal Fit')
    if best_mu is not None:
        ax0.text(0.55, 0.60, "Best-fit signal strength: {:.3f}".format(best_mu),
                transform=ax0.transAxes, fontsize=14)

    # Best fit and +/- 1, 2 sigma bands
    if G is not None or samples is not None:
        quantiles = [2.5, 16, 50, 84, 97.5]
        if samples is None:
            samples = [G.sample(x, num_samples=num_samples, 
                use_noise=use_noise) for x in binned['u']]
            bands = {q:[np.percentile(s, q) for s in samples] for q in quantiles}
        else:
            bands = {q:np.percentile(samples, q, axis=0) for q in quantiles}
        ax0.fill_between(centers, bands[2.5], bands[97.5], 
                facecolor='b', alpha=0.35)
        ax0.fill_between(centers, bands[16], bands[84], 
                facecolor='b', alpha=0.5)
        ax0.plot(centers, bands[50], color='b', label='GP Fit', linewidth=1.5)
        if samples_withsignal is not None:
            samples_withnoise = np.random.poisson(samples_withsignal)
        else:
            samples_withnoise = np.random.poisson(samples)
        nsigma = compute_nsigma(counts, samples_withnoise)
        ax1.bar(edges, nsigma, bin_width, align='edge', color='b',
                edgecolor='k')
        ax1.set_xlabel('MR (GeV)', fontsize=16)
        ax1.set_ylabel('Significance', fontsize=16)
        ax1.set_xlim(xmin=edges[0], xmax=edges[-1] + bin_width)
        ax1.set_ylim(ymin=-3.0, ymax=3.0)
        ax1.set_yticks(np.arange(-3, 4))
        ax1.tick_params(labelsize=14)
        ax1.set_axisbelow(True)
        ax1.yaxis.grid()
        plt.tight_layout()
    else:
        ax0.set_xlabel('MR (GeV)', fontsize=16)
    
    # Data counts
    ax0.errorbar(centers, counts, np.sqrt(counts), xerr=None, fmt='ko',
            label='Data')

    ax0.tick_params(labelsize=14)
    ax0.set_ylabel('Counts', fontsize=16)
    if title is not None:
        ax0.set_title(title, fontsize=16)
    if log:
        ax0.set_yscale('log')
    ax0.set_xlim(xmin=edges[0], xmax=edges[-1] + bin_width)
    ax0.set_ylim(ymin=ymin)
    ax0.legend(fontsize=14)

    if show:
        plt.show()

def plot_nsigma_1d(binned, G=None, samples=None, num_samples=40000,
        use_poisson_noise=False, verbose=False):
    """
    Inputs:
        binned: dataset loaded by Razor1DDataset or Razor2DDataset
        G: Gaussian Process object with a sample() method.
        samples: directly provide a list of samples 
        num_samples: number of samples to draw for each nsigma computation
        use_poisson_noise: only set to True if approximating Poisson noise
            using the observed data counts (for Gaussian GP likelihood)
    Plots the distribution of significances between data and fit.
    """
    centers = binned['u']
    counts = binned['y'].numpy()
    if samples is None:
        if use_poisson_noise:
            samples = [G.sample(x, num_samples=num_samples, 
                use_noise=True, 
                poisson_noise_vector=torch.Tensor([float(counts[i])])) 
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

