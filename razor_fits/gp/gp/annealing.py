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
        num_beta: roughly the number of annealing steps per run
        beta: list of annealing temperatures
        hmc: HamiltonianMC object implementing the jumps
        prior_sampler: function providing samples from the prior.
        num_hmc_samples: number of HMC samples to get after each annealing run
        samples: list of sampled parameter points
        log_importances: list of (log) importance weights
        last_beta: optionally end at this value of beta rather than beta=1.
            (used for optimizing the model)
        par_scheduler: optional function of one argument, providing beta 
            dependent HMC parameters.
    Note: don't try to use this with multiple parameter groups.
    """

    def __init__(self, hmc,
            num_beta, prior_sampler, num_hmc_samples=1, 
            par_scheduler=None, last_beta=None):
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
        self.par_scheduler = par_scheduler
        self.last_beta = last_beta
        self.samples = []
        self.log_importances = []
        self.accepted_steps = 0
        self.rejected_steps = 0

        # construct the annealing schedule
        # (note: leaves out beta = 0 and beta = 1,
        # which are handled separately).

        # Schedule inspired by the Neal paper.
        # Use an even spacing in log(beta), with several
        # ranges each with their own spacing value.
        # (note: this is very liberal with the definition of num_beta
        # and may give you many more betas than you ask for)
        # This was tuned for a particular box on num_beta = 1000
        ranges = [1e-40, 1e-30, 1e-20, 1e-10, 1e-6, 
                0.01, 0.05, 0.15, 0.3, 0.5, 0.85, 0.95, 1.0]
        ranges = np.log(np.asarray(ranges))
        num_betas = [num_beta for _ in ranges[:-1]]
        # adjust some ranges
        num_betas[0] = 0.1 * num_beta
        num_betas[-1] = 15.0 * num_beta 
        log_betas = []
        for i, (lower, upper) in enumerate(zip(ranges[:-1], ranges[1:])):
            num = num_betas[i]
            this_range = np.arange(lower, upper, (upper - lower)/num)
            log_betas.append(this_range)
        self.beta = np.exp(np.concatenate(log_betas))

    def get_prior_sample(self):
        sample = self.prior_sampler()
        for p, new_p in zip(self.params, sample):
            p.data.set_(new_p.data)

    def set_hmc_params(self, beta):
        if self.par_scheduler is not None:
            log_epsilon, L_max = self.par_scheduler(beta)
            epsilon = np.exp(log_epsilon)
            group = self.hmc.param_groups[0]
            group['epsilon'] = Variable(torch.Tensor([float(epsilon)]))
            group['L_max'] = Variable(torch.Tensor([int(L_max)]))

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
        last_jump = -1
        for i, beta in enumerate(self.beta):
            if self.last_beta is not None and beta >= self.last_beta:
                break
            this_closure = lambda: closure(beta)
            self.set_hmc_params(beta)
            initial_loss, final_loss = self.hmc.step(this_closure)
            initial_loss = float(initial_loss.data.numpy())
            final_loss = float(final_loss.data.numpy())
            log_w = log_w + final_loss - initial_loss
            if final_loss != initial_loss:
                last_jump = i
        # Get the final loss under the true likelihood
        final_loss = float(closure(1).data.numpy())
        log_w = log_w - final_loss
        self.samples.append([p.clone().data.numpy() for p in self.params])
        self.log_importances.append(log_w)
        if self.last_beta is None:
            print("log_w = {}".format(log_w))
            print("Last jump made: {} ({})".format(last_jump, 
        self.beta[last_jump]))
        # Save space by deleting the intermediate HMC samples
        self.hmc.samples = []
        if self.num_hmc_samples > 1:
            # Run regular HMC to get more samples
            beta_final = 1
            if self.last_beta is not None:
                beta_final = self.last_beta
            this_closure = lambda: closure(beta_final)
            self.set_hmc_params(beta_final)
            self.hmc.accepted_steps = 0
            self.hmc.rejected_steps = 0
            for i in range(self.num_hmc_samples - 1):
                self.hmc.step(this_closure)
            self.accepted_steps += self.hmc.accepted_steps
            self.rejected_steps += self.hmc.rejected_steps
            self.samples += self.hmc.samples

def run_annealing(G, parameters, num_runs, num_hmc_samples=1,
        num_beta=100, epsilon=0.0001, L_max=10, 
        verbose=True, print_every=10, par_scheduler=None,
        abort_on_error=False, last_beta=None):
    """
    Runs the annealed importance sampling procedure
    for the specified number of steps.
    The AnnealedImportanceSampler object is returned.
    """
    hmc = HamiltonianMC(parameters, epsilon, L_max,
            abort_on_error=abort_on_error)
    prior_sampler = G.sample_prior
    annealer = AnnealedImportanceSampler(hmc, num_beta, prior_sampler,
            num_hmc_samples, par_scheduler=par_scheduler, last_beta=last_beta)
    def closure(beta):
        hmc.zero_grad()
        G.clear()
        nlogp = G.neg_log_p(beta=beta)
        nlogp.backward()
        return nlogp
    for i in range(num_runs):
        annealer.step(closure)
    return annealer
