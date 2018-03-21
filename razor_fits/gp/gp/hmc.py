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
    """

    def __init__(self, params, epsilon=0.1, L_max=10, M=1.):
        defaults = dict(epsilon=epsilon, L_max=L_max, M=M)
        super(HamiltonianMC, self).__init__(params, defaults)
        self.samples = []

    def sample_phi(self, group):
        """
        Draw new values of the momenta conjugate to the parameters
        in the given group.
        """
        M = group['M']
        print("M", M)
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
        print("coeff", coeff)
        grads = self.get_grads(group)
        print("grads", grads)
        for p, g in zip(phi, grads):
            dphi = coeff * g
            print("dphi", dphi)
            p.data.add_(dphi.data)

    def theta_step(self, group, phi):
        coeff = group['epsilon'] / group['M']
        print("Coefficient", coeff)
        theta = group['params']
        for th, ph in zip(theta, phi):
            dtheta = coeff * ph
            print("dtheta", dtheta)
            th.data.add_(dtheta.data)

    def compute_r(self, initial_loss, final_loss, initial_phi, final_phi, 
            group):
        """Computes the accept-reject ratio r = p_final / p_initial."""
        M = group['M']
        log_p_phi = 0
        for phi_1, phi_L in zip(initial_phi, final_phi):
            log_p_phi += phi_1.pow(2).sum() - phi_L.pow(2).sum()
        log_p_phi = log_p_phi / M
        print("log_p_phi", log_p_phi)
        # initial_loss and final_loss are negative log p
        log_p_theta = initial_loss - final_loss
        print("log_p_theta", log_p_theta)
        return (log_p_theta + log_p_phi).exp().data.numpy()

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required")
        for group in self.param_groups:
            L = self.sample_L(group)
            print("L", L)
            phi = self.sample_phi(group)
            print("phi", phi)
            
            initial_theta = [t.clone() for t in group['params']]
            print("Initial theta", initial_theta)
            initial_phi = [p.clone() for p in phi]
            print("Initial phi", initial_phi)
            initial_loss = closure()
            print("Initial loss", initial_loss)
            self.phi_step(group, phi, half=True)
            print("After phi step", phi)
            for i in range(1, L+1):
                self.theta_step(group, phi)
                print("After theta step", group['params'])
                loss = closure()
                print("Loss", loss)
                self.phi_step(group, phi, half=(i == L))
                print("After phi step", phi)
            r = self.compute_r(initial_loss, loss, initial_phi, phi)
            print("Value of r", r)
            p = min(r, 1)
            print("Prob", p)
            if np.random.uniform() < p:
                # keep the new value
                print("Keeping!")
                sample = [t.clone().data.numpy() for t in group['params']]
            else:
                # reject the jump
                print("Rejecting...")
                sample = initial_theta
            samples.append(sample)


def run_hmc(G, num_samples, verbose=True, epsilon=0.1, L_max=10, clip=10):
    """
    First, the GP model is fit to obtain a reasonable estimate of the 
    parameter values. The learning rate specified by lr is used.
    Second, a HamiltonianMC object is created and is run for the specified
    number of iterations.
    The HamiltonianMC object is returned.
    """
    # TODO: multithread it
    pars = G.parameters()
    hmc = HamiltonianMC(pars, epsilon, L_max)
    def closure():
        hmc.zero_grad()
        G.clear()
        nlogp = G.neg_log_p()
        nlogp.backward()
        torch.nn.utils.clip_grad_norm(pars, clip)
        return nlogp
    for i in range(num_samples):
        hmc.step(closure)
    return hmc
