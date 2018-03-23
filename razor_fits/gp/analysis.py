import argparse
import time
import numpy as np
import torch
import torch.multiprocessing as mp

import razor_data
import gp

def get_data(box, btags, num_mr_bins, mr_max, proc='TTJets1L'):
    binned_data = razor_data.get_binned_data_1d(
            num_mr_bins=num_mr_bins, mr_max=mr_max, proc=proc)
    test_data = binned_data[box][btags]
    return test_data

### This function is called in parallel by fit() below
def _do_sample(seed, box, btags, num_mr_bins, mr_max, k_ell, k_alpha, 
        hmc_epsilon, hmc_L_max, num_samples, g, verbose=True):
    if seed >= 0:
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

    samples = G.sample(num_samples=num_samples, verbose=verbose)
    if seed >= 0:
        print("Seed {} done".format(seed))
    return samples

### This function runs the current best performing GP model 
### on a given analysis box.
def fit(
        box, btags,                                     # analysis region
        num_mr_bins, mr_max,                            # fit range 
        k_ell=200, k_alpha=200,                         # kernel parameters
        steps=1000, lr=0.001,                           # fit parameters
        hmc_epsilon=0.0001, hmc_L_max=10, chains=1,     # HMC parameters
        num_samples=40000, verbose=False, best_g=None,
        return_g=False): # specify return_g=True to return best value of g
    if best_g is None:
        # perform initial fit
        data = get_data(box, btags, num_mr_bins, mr_max)
        U = data['u']
        Y = data['y']
        kernel = gp.SquaredExponentialKernel(k_ell, k_alpha)
        G = gp.PoissonLikelihoodGP(kernel, U, Y, 
                hmc_epsilon=hmc_epsilon, hmc_L_max=hmc_L_max)
        G.fit(num_steps=steps, lr=lr, verbose=verbose)
        best_g = G.g.data
        if return_g: # we just want to fit and not sample
            return best_g

    n = int(num_samples / chains)
    args = [box, btags, num_mr_bins, mr_max, k_ell, k_alpha, 
            hmc_epsilon, hmc_L_max, n, best_g]
    if chains == 1:
        t0 = time.time()
        results = _do_sample(-1, *args)
        t1 = time.time()
        print("Total time: {}".format(t1 - t0))
        return results
    else:
        # Sample in parallel.
        # We have to use Pool.starmap to pass multiple arguments 
        # to the sampling function.
        args_list = [[i] + args for i in range(chains)]
        p = mp.Pool(chains)
        t0 = time.time()
        results = p.starmap(_do_sample, args_list)
        t1 = time.time()
        # Note: when I tested this I did not see any speedup
        # from parallelization.  
        print("Total time: {}".format(t1 - t0))
        return np.concatenate(results)

def bayes_opt(
        box, btags,                                     # analysis region
        num_mr_bins, mr_max,                            # fit range 
        k_ell=200, k_alpha=200,                         # kernel parameters
        steps=25, lr=0.001,                             # fit parameters
        num_samples=4000, verbose=True,
        iterations=40):
    """
    Performs Bayesian optimization of the tunable HMC parameters
    using the skopt package.  
    Optimization target is the negative log Expected Squared Jump Distance 
    (ESJD), which is the average squared distance traveled in each iteration.
    """
    import skopt
    print("Performing initial model fit")
    best_g = fit(box, btags, num_mr_bins, mr_max,
            k_ell, k_alpha, steps, lr, verbose=verbose, return_g=True)
    params = [
            (-11., -4.), # log(epsilon)
            (1, 30), # L_max
            ]
    optimizer = skopt.Optimizer(params)
    for i in range(iterations):
        next_params = optimizer.ask()
        next_log_epsilon, next_L_max = next_params
        if verbose:
            print("Beginning step {}: {:.2f} {}".format(i, 
                next_log_epsilon, next_L_max))
        samples = fit(box, btags, num_mr_bins, mr_max,
                k_ell, k_alpha, steps, lr, 
                float(np.exp(next_log_epsilon)), next_L_max,
                chains=1, num_samples=num_samples, verbose=verbose,
                best_g=best_g.clone())
        esjd = gp.compute_ESJD(samples, next_L_max)
        if esjd > 0:
            metric = -np.log(gp.compute_ESJD(samples, next_L_max))
        else:
            # -log is infinity
            metric = 9999
        if verbose:
            print("Obtained result -log(ESJD) = {}".format(metric))
        opt_result = optimizer.tell(next_params, metric)
        best_params = opt_result.x
        if verbose:
            print("New best parameter estimate: {}".format(best_params))
    print("Best HMC parameter estimate: {}".format(best_params))
    return best_params

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
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--verbose', action='store_true')
    return parser


if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = make_parser()
    a = parser.parse_args()
    result = bayes_opt(
        a.box, a.btags, a.num_mr_bins, a.mr_max,                           
        a.k_ell, a.k_alpha, a.steps, a.lr,                            
        a.samples, verbose=True)
