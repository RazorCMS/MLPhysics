import numpy as np
import torch
from torch.autograd import Variable

from .hmc import HamiltonianMC

class AnnealedImportanceSampler(torch.optim.Optimizer):
    """
    Implements annealed importance sampling (Neal, 1998)
    with Hamiltonian MC used for jump transitions between
    annealing steps.  
    Attributes:
        num_beta: number of annealing steps per run
        beta: list of annealing temperatures
        hmc: HamiltonianMC object implementing the jumps
        prior_sampler: function providing samples from the prior.
        num_hmc_samples: number of HMC samples to get after each annealing run
        samples: list of sampled parameter points
        importances: list of importance weights
    Note: don't try to use this with multiple parameter groups.
    """

    def __init__(self, hmc,
            num_beta, prior_sampler, num_hmc_samples=1):
        defaults = dict()
        params = hmc.param_groups
        super(AnnealedImportanceSampler, self).__init__(params, defaults)
        if len(self.param_groups) > 1:
            raise ValueError("This isn't designed for use with multiple groups")
        self.params = self.param_groups[0]['params']

        self.hmc = hmc
        self.num_beta = num_beta
        self.prior_sampler = prior_sampler
        self.num_hmc_samples = num_hmc_samples
        self.samples = []
        self.importances = []

        # construct the annealing schedule
        # (note: leaves out beta = 0 and beta = 1,
        # which are handled separately)
        d_beta = 1./num_beta
        self.beta = d_beta * np.arange(1, num_beta)

    def get_prior_sample(self):
        sample = self.prior_sampler()
        for p, new_p in zip(self.params, sample):
            p.data.set_(new_p.data)

    def step(self, closure=None):
        """
        Performs an annealing run.
        The closure should have the specific signature
        closure(beta) and 
        compute the loss at temperature beta.
        """
        if closure is None:
            raise ValueError("Closure required: closure(beta)")
        # Get initial sample from the prior and compute its loss
        self.get_prior_sample()
        log_w = float(closure(0).data.numpy()) # holds running importance weight
        for i, beta in enumerate(self.beta):
            if i % 100 == 0:
                print("At annealing step {} (beta = {:.6f})".format(
                    i, beta))
            this_closure = lambda: closure(beta)
            initial_loss, final_loss = self.hmc.step(this_closure)
            log_w = log_w + (float(final_loss.data.numpy() 
                - initial_loss.data.numpy()))
        # Get the final loss under the true likelihood
        log_w = log_w - float(closure(1).data.numpy())
        self.samples.append([p.clone().data.numpy() for p in self.params])
        self.importances.append(np.exp(log_w))
        # Save space by deleting the intermediate HMC samples
        self.hmc.samples = []
        if self.num_hmc_samples > 1:
            # Run regular HMC to get more samples
            print("Performing additional {} HMC steps".format(
                self.num_hmc_samples))
            this_closure = lambda: closure(1)
            for i in range(self.num_hmc_samples - 1):
                self.hmc.step(this_closure)
            self.samples += self.hmc.samples

def run_annealing(G, parameters, num_runs, num_hmc_samples=1,
        num_beta=100, epsilon=0.0001, L_max=10, 
        verbose=True, print_every=10,
        abort_on_error=False):
    """
    Runs the annealed importance sampling procedure
    for the specified number of steps.
    The AnnealedImportanceSampler object is returned.
    """
    hmc = HamiltonianMC(parameters, epsilon, L_max,
            abort_on_error=abort_on_error)
    prior_sampler = G.sample_prior
    annealer = AnnealedImportanceSampler(hmc, num_beta, prior_sampler,
            num_hmc_samples)
    def closure(beta):
        hmc.zero_grad()
        G.clear()
        nlogp = G.neg_log_p()
        nlogp.backward()
        return nlogp
    print("Beginning annealing runs with epsilon={}, L_max={}".format(
        epsilon, L_max))
    for i in range(num_runs):
        if verbose and i % print_every == 0 and i > 0:
            print("{}: Accepted steps: {}, Rejected steps: {} ({} bad)".format(
                i, hmc.accepted_steps, hmc.rejected_steps, hmc.bad_steps))
        annealer.step(closure)
    print("Variance of importance weights: {:.3f}".format(
        np.var(annealer.importances)))
    return annealer
