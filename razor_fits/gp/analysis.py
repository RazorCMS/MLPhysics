import argparse
import time
import numpy as np
import torch
import torch.multiprocessing as mp

import razor_data
import gp

def get_data(box, btags, num_mr_bins, mr_max):
    binned_data = razor_data.get_binned_data_1d(
            num_mr_bins=num_mr_bins, mr_max=mr_max)
    test_data = binned_data[box][btags]
    return test_data

### This function is called in parallel by fit() below
def _do_sample(seed, box, btags, num_mr_bins, mr_max, k_ell, k_alpha, 
        hmc_epsilon, hmc_L_max, num_samples, g, verbose=True):
    print("Sampling with seed {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = get_data(box, btags, num_mr_bins, mr_max)
    U = data['u']
    Y = data['y']
    kernel = gp.SquaredExponentialKernel(k_ell, k_alpha)
    G = gp.PoissonLikelihoodGP(kernel, U, Y, 
            hmc_epsilon=hmc_epsilon, hmc_L_max=hmc_L_max)
    # input g should be a torch Tensor
    G.g = torch.nn.Parameter(g.clone())

    G.sample(num_samples=num_samples, verbose=verbose)
    print("Seed {} done".format(seed))
    return G.samples

### This function runs the current best performing GP model 
### on a given analysis box.
def fit(
        box, btags,                                     # analysis region
        num_mr_bins, mr_max,                            # fit range 
        k_ell=200, k_alpha=200,                         # kernel parameters
        steps=25, lr=0.001,                             # fit parameters
        hmc_epsilon=0.0001, hmc_L_max=10, chains=1,     # HMC parameters
        num_samples=40000, verbose=False):
    # perform initial fit
    data = get_data(box, btags, num_mr_bins, mr_max)
    U = data['u']
    Y = data['y']
    kernel = gp.SquaredExponentialKernel(k_ell, k_alpha)
    G = gp.PoissonLikelihoodGP(kernel, U, Y, 
            hmc_epsilon=hmc_epsilon, hmc_L_max=hmc_L_max)
    G.fit(num_steps=steps, lr=lr, verbose=verbose)
    best_g = G.g.data

    # sample in parallel
    n = int(num_samples / chains)
    args = [box, btags, num_mr_bins, mr_max, k_ell, k_alpha, 
            hmc_epsilon, hmc_L_max, n, best_g]
    if chains == 1:
        t0 = time.time()
        results = _do_sample(1, *args)
        t1 = time.time()
        print("Total time: {}".format(t1 - t0))
        return results
    else:
        # We have to use Pool.starmap to pass multiple arguments 
        # to the sampling function.
        args_list = [[i] + args for i in range(chains)]
        p = mp.Pool(chains)
        t0 = time.time()
        results = p.starmap(_do_sample, args_list)
        t1 = time.time()
        print("Total time: {}".format(t1 - t0))
        return np.concatenate(results)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('box')
    parser.add_argument('btags', type=int)
    parser.add_argument('--num-mr-bins', type=int, default=50)
    parser.add_argument('--mr-max', type=int, default=1200)
    parser.add_argument('--k-ell', type=float, default=200)
    parser.add_argument('--k-alpha', type=float, default=200)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hmc-epsilon', type=float, default=0.0001)
    parser.add_argument('--hmc-L-max', type=int, default=10)
    parser.add_argument('--chains', type=int, default=1)
    parser.add_argument('--samples', type=int, default=40000)
    parser.add_argument('--verbose', action='store_true')
    return parser


if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = make_parser()
    a = parser.parse_args()
    samples = fit(a.box, a.btags, a.num_mr_bins, a.mr_max,
            a.k_ell, a.k_alpha, a.steps, a.lr, 
            a.hmc_epsilon, a.hmc_L_max, a.chains,
            a.samples, a.verbose)
    print(samples)
