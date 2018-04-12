import argparse
import time
import bisect
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp

import razor_data
import plotting
import gp

def get_data(box, btags, num_mr_bins, mr_max, proc='TTJets1L', scale=1.0):
    if proc.lower() == 'fake':
        binned_data = razor_data.get_binned_data_1d(
                num_mr_bins=num_mr_bins, mr_max=mr_max)
        test_data = binned_data[box][btags]
        U = test_data['u'].numpy()
        bin_width = U[1] - U[0]
        # Gaussian resonance 
        mu = 1000
        sigma = 50
        norm = 200
        test_data['y'] = norm * bin_width * torch.Tensor(
                plotting.gauss(U, mu, sigma))
    else:
        binned_data = razor_data.get_binned_data_1d(
                num_mr_bins=num_mr_bins, mr_max=mr_max, proc=proc)
        test_data = binned_data[box][btags]
        test_data['u'] = test_data['u'] * scale
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

def fit(
        box, btags,                                     # analysis region
        num_mr_bins, mr_max,                            # fit range 
        k_ell=200, k_alpha=200,                         # kernel parameters
        steps=1000, lr=0.001,                           # fit parameters
        hmc_epsilon=0.0001, hmc_L_max=10, chains=1,     # HMC parameters
        num_samples=40000, verbose=False, best_g=None,
        return_g=False): # specify return_g=True to return best value of g
    """
    Fits the GP model (with no signal) on the chosen analysis box
    and returns samples from the GP posterior.
    """
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

def fit_signal(
        box, btags, sms,                                # analysis region
        num_mr_bins, mr_max,                            # fit range
        k_ell=200, k_alpha=200,                         # kernel parameters
        steps=1000, lr=0.001,                           # fit parameters
        hmc_epsilon=0.0001, hmc_L_max=10,               # HMC parameters
        num_samples=40000, 
        mu_true=1.0,                                    # signal strength
        best_pars=None, return_pars=False,                    
        verbose=False,
        kernel_gp=None, scale=1.0): 
    """
    Performs a signal + background fit using a fixed signal shape.
    Samples from the GP posterior and returns the GP object.
    Access the samples via G.preds_from_samples().
    (note: gave up on multiprocessing for now.)
    """
    data = get_data(box, btags, num_mr_bins, mr_max, scale=scale)
    data_sig = get_data(box, btags, num_mr_bins, mr_max, 
                             proc=sms, scale=scale)
    U = data['u']
    Y = data['y']
    # inject signal into the data
    S_mean = data_sig['y']
    true_signal = np.random.poisson(S_mean.numpy() * mu_true)
    Y = Y + torch.Tensor(true_signal)
    
    if kernel_gp is None:
        kernel = gp.SquaredExponentialKernel(k_ell, k_alpha, fixed=True)
    else:
        kernel = kernel_gp
    print(kernel.forward(Variable(U), Variable(U)))
    G = gp.PoissonGPWithSignal(kernel, U, Y, S_mean,
            hmc_epsilon=hmc_epsilon, hmc_L_max=hmc_L_max,
            num_true_signal=true_signal.sum())
    if best_pars is None:
        G.fit(num_steps=steps, lr=lr, verbose=verbose)
        best_pars = (G.g.data, G.signal.data)
        if return_pars:
            return best_pars
    G.g = torch.nn.Parameter(best_pars[0].clone())
    G.signal = torch.nn.Parameter(best_pars[1].clone())
    G.sample(num_samples, verbose=verbose)

    return G

def bayes_opt(
        box, btags,                                     # analysis region
        num_mr_bins, mr_max,                            # fit range 
        k_ell=200, k_alpha=200,                         # kernel parameters
        steps=1000, lr=0.001,                           # fit parameters
        num_samples=500, verbose=True,
        iterations=40, penalize_L=True,
        sms=None, mu_true=None,                         # optional signal
        kernel_gp=None, scale=1.0):                        
    """
    Performs Bayesian optimization of the tunable HMC parameters
    using the skopt package.  
    Optimization target is the negative log Expected Squared Jump Distance 
    (ESJD), which is the average squared distance traveled in each iteration.
    """
    import skopt
    params = [
            (-11., -2.), # log(epsilon)
            (1, 30), # L_max
            ]
    print("Performing initial model fit")
    if sms is None:
        best_g = fit(box, btags, num_mr_bins, mr_max,
                k_ell, k_alpha, steps, lr, verbose=verbose, return_g=True)
    else:
        best_g, best_signal = fit_signal(box, btags, sms,
                num_mr_bins, mr_max, k_ell=k_ell, k_alpha=k_alpha, 
                steps=steps, lr=lr,
                mu_true=mu_true, return_pars=True, verbose=verbose,
                kernel_gp=kernel_gp, scale=scale)
    optimizer = skopt.Optimizer(params)
    for i in range(iterations):
        next_params = optimizer.ask()
        next_log_epsilon, next_L_max = next_params
        if verbose:
            print("Beginning step {}: {:.2f} {}".format(i, 
                next_log_epsilon, next_L_max))
        if sms is None:
            samples = fit(box, btags, num_mr_bins, mr_max,
                    k_ell, k_alpha, steps, lr, 
                    float(np.exp(next_log_epsilon)), next_L_max,
                    chains=1, num_samples=num_samples, verbose=verbose,
                    best_g=best_g.clone())
        else:   
            G = fit_signal(box, btags, sms, num_mr_bins, mr_max,
                    k_ell=k_ell, k_alpha=k_alpha, steps=steps, lr=lr, 
                    hmc_epsilon=float(np.exp(next_log_epsilon)), 
                    hmc_L_max=next_L_max,
                    num_samples=num_samples, mu_true=mu_true,
                    best_pars=(best_g, best_signal), verbose=verbose,
                    kernel_gp=kernel_gp, scale=scale)
            samples = G.samples
        esjd = gp.compute_ESJD(samples, next_L_max, penalize=penalize_L)
        if esjd > 0:
            metric = -np.log(esjd)
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

def run_s_plus_b(box, btags, sms,                       # analysis region
        num_mr_bins, mr_max,                            # fit range 
        k_ell=200, k_alpha=200,                         # kernel parameters
        steps=1000, lr=0.001,                           # fit parameters
        num_opt_samples=500, 
        opt_iterations=30, penalize_L=True,
        mu_true=1.0, num_samples=160000,
        runs=1, verbose=True): 
    print("Will first run Bayesian optimization "
          "for {} iterations "
          "on the HMC hyperparameters".format(opt_iterations))
    best_hmc_log_epsilon, best_hmc_L_max = bayes_opt(
            box, btags, num_mr_bins, mr_max,
            k_ell, k_alpha, steps, lr,
            num_opt_samples, verbose, 
            opt_iterations, penalize_L,
            sms, mu_true)
    best_hmc_epsilon = float(np.exp(best_hmc_log_epsilon))
    print("Will do {} runs of {} samples "
            "with epsilon = {:.6f}, L_max = {}".format
          (runs, num_samples, best_hmc_epsilon, best_hmc_L_max))
    results = []
    for i in range(runs):
        print("Starting run #{}".format(i))
        results.append(fit_signal(box, btags, sms,
            num_mr_bins, mr_max, k_ell, k_alpha, steps, lr, 
            best_hmc_epsilon, best_hmc_L_max,
            num_samples, mu_true, verbose=verbose))
        G = results[-1]
        samples = G.preds_from_samples()
        samples_withsignal = G.preds_from_samples(include_signal=True)
        best_mu = np.percentile([s[0] for s in G.signal_samples], 
                50)
        num_fitted_signal = best_mu * G.S.data.numpy().sum()
        num_true_signal = G.num_true_signal
        out_name = "{}_{}b_{}_mu-{}_run-{}.png".format(
            box, btags, sms, mu_true, i)
        plotting.plot_hist_1d(U=G.U.data, Y=G.Y.data, S=G.S.data * mu_true,
            samples=samples, samples_withsignal=samples_withsignal, show=False,
            best_mu=best_mu)
        plt.savefig(out_name)
        print("Best-fit signal: {:.3f}".format(best_mu))
        print("Fitted {:.2f} signal events vs. {:.2f} generated".format(
            num_fitted_signal, num_true_signal))
        print("Wrote image file {}".format(out_name))
    return results

def fit_interpolation(
        box, btags,                                     # analysis region
        num_mr_bins, mr_max,                            # fit range 
        interp_bins,                                    # bins to predict
        k_ell=200, k_alpha=200,                         # kernel parameters
        steps=1000, lr=0.001,                           # fit parameters
        hmc_epsilon=0.0001, hmc_L_max=10,               # HMC parameters
        num_samples=40000, verbose=False): 
    """
    Fits the GP model (with no signal) on the chosen analysis box
    and interpolates to make predictions for blinded bins
    """
    kernel = gp.SquaredExponentialKernel(k_ell, k_alpha)
    data = get_data(box, btags, num_mr_bins, mr_max)
    U = data['u']
    Y = data['y']
    
    fit_index = [i for i in range(U.size(0)) if i not in interp_bins]
    fit_U = U[fit_index]
    fit_Y = Y[fit_index]

    # perform initial fit
    G = gp.PoissonLikelihoodGP(kernel, fit_U, fit_Y, 
            hmc_epsilon=hmc_epsilon, hmc_L_max=hmc_L_max)
    G.fit(num_steps=steps, lr=lr, verbose=verbose)
    G.sample(num_samples=num_samples, verbose=verbose)
    return G

def run_interpolation(
        box, btags,                                     # analysis region
        num_mr_bins, mr_max,                            # fit range 
        interp_bins,                                    # bins to predict
        k_ell=200, k_alpha=200,                         # kernel parameters
        steps=1000, lr=0.001,                           # fit parameters
        hmc_epsilon=0.0001, hmc_L_max=10,               # HMC parameters
        num_samples=40000, verbose=False): 
    G = fit_interpolation(box, btags, num_mr_bins, mr_max,
        interp_bins, k_ell, k_alpha, steps, lr, hmc_epsilon, hmc_L_max,
        num_samples, verbose) 

    data = get_data(box, btags, num_mr_bins, mr_max)
    U = data['u']
    Y = data['y']
    num_bins = U.size(0)
    fit_bins = [i for i in range(num_bins) if i not in interp_bins]
    V = U[interp_bins]
    Y[interp_bins] = 0. # mask interpolation bins

    fit_preds = G.preds_from_samples()
    interp_preds = G.predict(V)
    all_preds = np.zeros((num_samples, num_bins))
    all_preds[:, interp_bins] = interp_preds
    all_preds[:, fit_bins] = fit_preds
    plotting.plot_hist_1d(U=U, Y=Y, samples=all_preds, show=False)
    out_name = "interpolation_{}_{}b.png".format(box, btags)
    plt.savefig(out_name)
    print("Wrote image file {}".format(out_name))

    return G

def fit_annealer(box, btags, num_mr_bins, mr_max, 
        sms, mu_true=1.0, mu_test=1.0,
	k_ell=200, k_alpha=200,
        par_scheduler=None, last_beta=None,
	num_runs=1, num_hmc_steps=1, num_beta=100,
	verbose=False, best_params=None):
    data = get_data(box, btags, num_mr_bins, mr_max)
    data_sig = get_data(box, btags, num_mr_bins, mr_max, sms)
    kernel = gp.SquaredExponentialKernel(k_ell, k_alpha)
    U = data['u']
    Y = data['y']
    S_mean = data_sig['y']
    true_signal = np.random.poisson(S_mean.numpy() * mu_true)
    Y = Y + torch.Tensor(true_signal)

    # parameter scheduler based on best params from BayesOpt
    if par_scheduler is None:
        if best_params is None:
            raise ValueError(
                    "Please provide HMC parameters or a scheduler function")
        betas = sorted([b for b in best_params])
        def scheduler(beta):
            if beta <= 0:
                return best_params[0]
            if beta >= 1:
                return best_params[1]
            # find the two beta nodes we are between
            beta_index = bisect.bisect_left(betas, beta)
            beta_low = betas[beta_index - 1]
            beta_high = betas[beta_index]
            frac = (beta - beta_low) / (beta_high - beta_low)
            # interpolate between parameters
            log_eps_low, K_max_low = best_params[beta_low]
            log_eps_high, K_max_high = best_params[beta_high]
            log_eps = log_eps_low + frac * (
                    log_eps_high - log_eps_low)
            K_max = K_max_low + frac * (K_max_high - K_max_low)
            #ret = float(log_eps_low), int(K_max_low)
            ret = float(log_eps), int(K_max)
            return ret
        par_scheduler = scheduler

    G = gp.AnnealingPoissonGP(kernel, U, Y, S_mean, mu=mu_test, 
            hmc_par_scheduler=par_scheduler,
            num_true_signal=true_signal.sum())
    G.sample(num_runs, num_hmc_steps, num_beta, 
             verbose=verbose, print_every=100, last_beta=last_beta)
    return G

def bayes_opt_annealing(
        box, btags,                                     # analysis region
        num_mr_bins, mr_max,                            # fit range 
        sms, mu_true=1.0, mu_test=1.0,                  # signal
        k_ell=200, k_alpha=200,                         # kernel parameters
        num_runs=100, num_hmc_steps=100, num_beta=1000, # annealing parameters
        iterations=10, verbose=True, precomputed_pars={},
        use_acceptance_prob=False):
    """
    For the annealed importance sampling algorithm we need to 
    make good HMC jumps at different annealing temperatures.  
    The final likelihood is much sharper than the prior so different
    step sizes will be necessary at each stage. 
    We will test the HMC jump distance at several values of temperature,
    letting epsilon change smoothly from node to node.  
    This function performs Bayesian optimization on the HMC schedule.
    We will keep the number of leapfrog steps small to avoid
    very slow annealing.
    """
    import skopt
    params = [
            (-11., 0.), # log(epsilon)
            (1, 10), # L_max
            ]
    betas_to_optimize = [0, 1e-10, 1e-5, 0.01, 0.05, 0.3, 1.0]
    best_params = {}
    for cur_beta in betas_to_optimize:
        if cur_beta in precomputed_pars:
            print("Best params for beta = {} are precomputed: {}".format(
                cur_beta, precomputed_pars[cur_beta]))
            best_params[cur_beta] = precomputed_pars[cur_beta]
            continue
        optimizer = skopt.Optimizer(params)
        for i in range(iterations):
            next_params = optimizer.ask()
            next_log_epsilon, next_L_max = next_params
            if verbose:
                print("Beginning step {}: {:.2f} {}".format(i, 
                    next_log_epsilon, next_L_max))
            # temporarily put these in here
            best_params[cur_beta] = (float(next_log_epsilon), 
                    int(next_L_max))
            # The annealer uses this function to adapt the HMC parameters
            # on the fly as beta changes
            def par_scheduler(beta):
                if beta <= 0:
                    return best_params[0]
                if beta >= 1:
                    return best_params[1]
                # find the two beta nodes we are between
                beta_index = bisect.bisect_left(betas_to_optimize, beta)
                beta_low = betas_to_optimize[beta_index - 1]
                beta_high = betas_to_optimize[beta_index]
                frac = (beta - beta_low) / (beta_high - beta_low)
                # interpolate between parameters
                log_eps_low, K_max_low = best_params[beta_low]
                log_eps_high, K_max_high = best_params[beta_high]
                log_eps = log_eps_low + frac * (
                        log_eps_high - log_eps_low)
                K_max = K_max_low + frac * (K_max_high - K_max_low)
                ret = float(log_eps), int(K_max)
                return ret
            G = fit_annealer(box, btags, num_mr_bins, mr_max,
                    sms, mu_true, mu_test,
                    k_ell=k_ell, k_alpha=k_alpha, 
                    par_scheduler=par_scheduler, last_beta=cur_beta,
                    num_runs=num_runs, num_hmc_steps=num_hmc_steps,
                    num_beta=num_beta, verbose=verbose)
            if use_acceptance_prob:
                # evaluate based on jump acceptance probability
                target_prob = 0.63
                actual_prob = G.annealer.accepted_steps/(
                        G.annealer.accepted_steps + G.annealer.rejected_steps)
                metric = (target_prob - actual_prob)**2
                if verbose:
                    print("Obtained squared loss {:.5f} (prob = {:.3f}".format(
                        metric, actual_prob))
            else:
                # compute ESJD for each run and average them
                samples = G.samples
                esjds = []
                for r in range(num_runs):
                    start = r * num_hmc_steps
                    end = start + num_hmc_steps
                    r_samples = samples[start:end]
                    esjds.append(gp.compute_ESJD(r_samples, next_L_max))
                esjd = np.mean(esjds)
                # try: multiply by acceptance probability
                accept_prob = G.annealer.accepted_steps/(
                        G.annealer.accepted_steps + G.annealer.rejected_steps)
                if esjd > 0 and accept_prob > 0:
                    metric = -np.log(esjd * accept_prob / np.sqrt(next_L_max))
                else:
                    # -log is infinity
                    metric = 9999
                if verbose:
                    print("Obtained result -log(ESJD) = {}".format(metric))
            opt_result = optimizer.tell(next_params, metric)
        best_params[cur_beta] = opt_result.x
        if verbose:
            print("Best param estimate for beta={}: {}".format(
                cur_beta, best_params[cur_beta]))
    print("Best HMC parameter estimates: {}".format(best_params))
    return best_params

def run_signal_scan(
        box, btags, sms,                                # analysis region
        num_mr_bins, mr_max,                            # fit range
        k_ell=200, k_alpha=200,                         # kernel parameters
        steps=1000, lr=0.001,                           # fit parameters
        mu_min=-1.0, mu_max=5.0, mu_step=0.05, mu_true=1.0,
        verbose=False): 
    """
    Scans the signal strength from mu_min to mu_max.
    At each point, performs a fit and gets the resulting GP.
    """
    data = get_data(box, btags, num_mr_bins, mr_max)
    data_sig = get_data(box, btags, num_mr_bins, mr_max, 
                             proc=sms)
    U = data['u']
    # fix the random signal realization for all runs
    S_mean = data_sig['y']
    true_signal = np.random.poisson(S_mean.numpy() * mu_true)
    Y = data['y'] + torch.Tensor(true_signal)
    kernel = gp.SquaredExponentialKernel(k_ell, k_alpha, fixed=True)
    Gs = []
    for mu in np.arange(mu_min, mu_max, mu_step):
        if verbose:
            print("Fitting GP with signal = {:.3f}".format(mu))
        G = gp.PoissonGPWithSignal(kernel, U, Y, S_mean,
                num_true_signal=true_signal.sum())
        G.signal = torch.nn.Parameter(torch.Tensor([mu]), requires_grad=False)
        G.fit(num_steps=steps, lr=lr, verbose=False,
                parameters=[G.g])
        Gs.append(G)
    return Gs

def run_s_plus_b_deepkernel(box, btags, sms,            # analysis region
        num_mr_bins, mr_max,                            # fit range 
        kernel_gp,                                      # deep GP to use as kernel
        steps=1000, lr=0.001,                           # parameters for initial fit
        num_opt_samples=500, 
        opt_iterations=30, penalize_L=True,
        mu_true=1.0, num_samples=160000,
        runs=1, verbose=True, scale=0.001): 
    print("Will first run Bayesian optimization "
          "for {} iterations "
          "on the HMC hyperparameters".format(opt_iterations))
    best_hmc_log_epsilon, best_hmc_L_max = bayes_opt(
            box, btags, num_mr_bins, mr_max,
            steps=steps, lr=lr, kernel_gp=kernel_gp,
            num_samples=num_opt_samples, verbose=verbose, 
            iterations=opt_iterations, penalize_L=penalize_L,
            sms=sms, mu_true=mu_true, scale=scale)
    best_hmc_epsilon = float(np.exp(best_hmc_log_epsilon))
    print("Will do {} runs of {} samples "
            "with epsilon = {:.6f}, L_max = {}".format
          (runs, num_samples, best_hmc_epsilon, best_hmc_L_max))
    results = []
    for i in range(runs):
        print("Starting run #{}".format(i))
        results.append(fit_signal(box, btags, sms,
            num_mr_bins, mr_max, 
            kernel_gp=kernel_gp, steps=steps, lr=lr, 
            best_hmc_epsilon=best_hmc_epsilon, 
            best_hmc_L_max=best_hmc_L_max,
            num_samples=num_samples, mu_true=mu_true, verbose=verbose,
            scale=scale))
        G = results[-1]
        samples = G.preds_from_samples()
        samples_withsignal = G.preds_from_samples(include_signal=True)
        best_mu = np.percentile([s[0] for s in G.signal_samples], 
                50)
        num_fitted_signal = best_mu * G.S.data.numpy().sum()
        num_true_signal = G.num_true_signal
        out_name = "{}_{}b_{}_mu-{}_run-{}.png".format(
            box, btags, sms, mu_true, i)
        plotting.plot_hist_1d(U=G.U.data, Y=G.Y.data, S=G.S.data * mu_true,
            samples=samples, samples_withsignal=samples_withsignal, show=False,
            best_mu=best_mu, x_scale=scale)
        plt.savefig(out_name)
        print("Best-fit signal: {:.3f}".format(best_mu))
        print("Fitted {:.2f} signal events vs. {:.2f} generated".format(
            num_fitted_signal, num_true_signal))
        print("Wrote image file {}".format(out_name))
    return results


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
    parser.add_argument('--no-penalize', action='store_true')
    return parser


if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = make_parser()
    a = parser.parse_args()
    result = bayes_opt(
        a.box, a.btags, a.num_mr_bins, a.mr_max,                           
        a.k_ell, a.k_alpha, a.steps, a.lr,                            
        a.samples, verbose=True, penalize_L=(not a.no_penalize))
