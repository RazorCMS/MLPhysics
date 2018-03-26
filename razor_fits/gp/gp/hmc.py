import numpy as np
import torch
from torch.autograd import Variable

class HamiltonianMC(torch.optim.Optimizer):
    """
    Implements the Hamiltonian Monte Carlo sampling algorithm.  
    Attributes:
        epsilon: step size
        L_max: maximum number of leapfrog steps per iteration.
            At each iteration, a number of steps L is drawn uniformly
            between 1 and L_max.
        samples: list of sampled parameter points
        abort_on_error: give up if Torch throws a runtime error.
            (usually they are overflows from large gradients
             when dealing with sharply peaked likelihoods)
    """

    def __init__(self, params, epsilon=0.1, L_max=10, M=1.,
            abort_on_error=False):
        defaults = dict(epsilon=epsilon, L_max=L_max, M=M)
        super(HamiltonianMC, self).__init__(params, defaults)
        self.abort_on_error = abort_on_error
        self.samples = []
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.bad_steps = 0

    def sample_phi(self, group):
        """
        Draw new values of the momenta conjugate to the parameters
        in the given group.
        """
        M = group['M']
        return [Variable(torch.randn(p.size(0))) * 
                Variable(torch.Tensor([np.sqrt(M)]))
                for p in group['params']]

    def sample_L(self, group):
        """
        Randomly draw a number of leapfrog steps between 1 and L_max.
        """
        L_max = group['L_max']
        return np.random.randint(1, L_max + 1)

    def get_grads(self, group):
        """
        Extracts the gradients of the parameters in the given group.
        """
        return [p.grad for p in group['params']]

    def phi_step(self, group, phi, half=False):
        coeff = group['epsilon']
        if half:
            coeff *= 0.5
        grads = self.get_grads(group)
        for p, g in zip(phi, grads):
            dphi = coeff * g
            p.data.add_(dphi.data)

    def theta_step(self, group, phi):
        coeff = group['epsilon'] / group['M']
        theta = group['params']
        for th, ph in zip(theta, phi):
            dtheta = coeff * ph
            th.data.add_(dtheta.data)

    def compute_r(self, initial_loss, final_loss, initial_phi, final_phi, 
            group):
        """Computes the accept-reject ratio r = p_final / p_initial."""
        M = group['M']
        log_p_phi = 0
        for phi_1, phi_L in zip(initial_phi, final_phi):
            try:
                log_p_phi += phi_1.pow(2).sum() - phi_L.pow(2).sum()
            except RuntimeError:
                # This may occasionally overflow if phi gets too large.
                # It will only happen for the final phi value (the initial
                # value will not be this large).  Treat this as p(phi) = 0
                # and therefore r = 0.
                if self.abort_on_error:
                    raise
                return 0
        log_p_phi = log_p_phi / M
        # initial_loss and final_loss are negative log p
        log_p_theta = initial_loss - final_loss
        return (log_p_theta + log_p_phi).exp().data.numpy()

    def clone_params(self, group):
        return [p.clone() for p in group['params']]

    def reset_theta(self, group, initial_theta):
        """
        Sets the group params to copies of the parameters in initial_theta.
        """
        # even if initial_theta is already a clone of the original
        # GP parameters, we have to clone it again to avoid the HMC samples
        # and the output GP parameters sharing memory.
        cloned = [p.clone() for p in initial_theta]
        for p, new_p in zip(group['params'], cloned):
            p.data.set_(new_p.data) 

    def append_sample(self, sample):
        self.samples.append([p.data.numpy() for p in sample])

    def accept_jump(self, group):
        """
        Accepts the current HMC jump.
        """
        self.append_sample(self.clone_params(group))
        self.accepted_steps +=1 

    def reject_jump(self, group, initial_theta):
        """
        Rejects the current HMC jump.
        Returns the next HMC sample as a list of Tensors.
        """
        self.reset_theta(group, initial_theta)
        self.append_sample(initial_theta)
        self.rejected_steps += 1

    def step(self, closure=None):
        """
        Note: returns the values of the loss at the beginning
        and end of the step (for use with annealed importance sampling)
        """
        if closure is None:
            raise ValueError("Closure required")
        for group in self.param_groups:
            # note: I haven't tested this with multiple parameter groups.
            # It might be buggy or require unexpected semantics
            # (for sure, the list of samples will need to be sliced 
            # correctly to get the samples corresponding to each group)
            L = self.sample_L(group)
            phi = self.sample_phi(group)
            
            initial_theta = self.clone_params(group)
            initial_phi = [p.clone() for p in phi]
            try:
                initial_loss = closure()
            except RuntimeError:
                print(
                    "The loss function threw an error on the first evaluation"
                    " -- aborting.")
                if self.abort_on_error:
                    raise
                self.bad_steps += 1
                self.reject_jump(group, initial_theta)
                return initial_loss, initial_loss
            self.phi_step(group, phi, half=True)
            for i in range(1, L+1):
                self.theta_step(group, phi)
                try:
                    loss = closure()
                except RuntimeError:
                    # Errors in the loss calculation are caused by overflows
                    # due to very large gradients.  If we encounter this,
                    # give up and abort the step.  
                    if self.abort_on_error:
                        raise
                    self.bad_steps += 1
                    self.reject_jump(group, initial_theta)
                    return initial_loss, initial_loss
                # The last step is a half-step in phi; all other steps are full.
                self.phi_step(group, phi, half=(i == L))
            r = self.compute_r(initial_loss, loss, initial_phi, phi, group)
            p = min(r, 1)
            if np.random.uniform() < p:
                self.accept_jump(group)
                return initial_loss, loss
            else:
                self.reject_jump(group, initial_theta)
                return initial_loss, initial_loss

    def get_accept_rate(self):
        return self.accepted_steps / (self.accepted_steps + self.rejected_steps)


def run_hmc(G, parameters, num_samples, verbose=True, 
        epsilon=0.1, L_max=10, print_every=10, 
        abort_on_error=False):
    """
    First, the GP model is fit to obtain a reasonable estimate of the 
    parameter values. The learning rate specified by lr is used.
    Second, a HamiltonianMC object is created and is run for the specified
    number of iterations.
    The HamiltonianMC object is returned.
    """
    hmc = HamiltonianMC(parameters, epsilon, L_max,
            abort_on_error=abort_on_error)
    def closure():
        hmc.zero_grad()
        G.clear()
        nlogp = G.neg_log_p()
        nlogp.backward()
        return nlogp
    print("Beginning HMC iterations with epsilon={}, L_max={}".format(
        epsilon, L_max))
    for i in range(num_samples):
        if verbose and i % print_every == 0 and i > 0:
            print("{}: Accepted steps: {}, Rejected steps: {} ({} bad)".format(
                i, hmc.accepted_steps, hmc.rejected_steps, hmc.bad_steps))
        hmc.step(closure)
    print("Acceptance rate: {:.3f} ({} bad samples)".format(
        hmc.get_accept_rate(), hmc.bad_steps))
    return hmc

def compute_ESJD(samples, L=1, penalize=True):
    """
    Compute the expected squared jumping distance (ESJD), optionally 
    penalized by the square root of the number of HMC leapfrog steps L.
    Input should be a (S x d) numpy array, with S the number of MCMC samples.
    """
    diff = samples[1:] - samples[:-1]
    esjd = np.power(diff, 2).sum() / samples.shape[0]
    if penalize:
        return esjd / np.sqrt(L)
    return esjd
