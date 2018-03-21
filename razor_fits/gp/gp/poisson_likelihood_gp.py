import numpy as np
import torch
from torch.autograd import Variable

from .gaussian_process import GaussianProcess
from .kernel import make_log_par

class PoissonLikelihoodGP(GaussianProcess):
    """
    GP with a Poisson conditional likelihood.
    Some computation results are memoized internally in order
    to speed up prediction.  Clear the memoized values with
    clear() before e.g. using the GP with new kernel hyperparameters.
    Attributes:
        mu: placeholder for the prior mean.
        K: placeholder for the covariance matrix K.
        epsilon: small value added to diagonal of K for numerical stability
        raw_K: K computed by the kernel function, before adding epsilon
        L: Cholesky decomposition of K
        f: best-fit function values at each input point
        g: best-fit function values stored in whitened form (i.e. f = Lg)
    """

    def __init__(self, kernel, U, Y, mean=None):
        super(PoissonLikelihoodGP, self).__init__(kernel, U, Y, mean)
        self.mu = None
        self.K = None
        self.epsilon = None
        self.raw_K = None
        self.L = None 
        self.f = None 

        # Initialize to observed data positions
        # i.e. Lg = log(Y)
        L = self.get_L()
        initial_g = torch.gesv((Y+0.1).log(), L.data)[0].squeeze()
        self.g = torch.nn.Parameter(initial_g)
        self.clear()

    def clear(self):
        self.mu = None
        self.K = None
        self.epsilon = None
        self.raw_K = None
        self.L = None
        self.f = None

    def get_mu(self):
        if self.mu is not None:
            return self.mu
        self.mu = self.mean(self.U)
        return self.mu

    def get_raw_K(self):
        if self.raw_K is not None:
            return self.raw_K
        U = self.U
        self.raw_K = self.kernel.forward(U, U)
        return self.raw_K

    def get_epsilon(self):
        if self.epsilon is not None:
            return self.epsilon
        epsilon = 1e-5
        K = self.get_raw_K()
        min_eig = torch.min(torch.eig(K)[0][:,0]).data.numpy()[0]
        if min_eig < 0:
            if min_eig < -0.1:
                print("Warning: min eigenvalue is negative: {}".format(min_eig))
            epsilon = -10 * min_eig
        self.epsilon = Variable(torch.Tensor([epsilon]))
        return self.epsilon

    def get_K(self):
        """
        Gets the covariance matrix with a small multiple of the identity
        added to prevent numerical instabilities in the Cholesky matrix
        computation.
        """
        if self.K is not None:
            return self.K
        U = self.U
        K = self.get_raw_K()
        epsilon = self.get_epsilon()
        noise = epsilon * Variable(torch.eye(U.size(0)))
        self.K = K + noise
        return self.K

    def get_L(self):
        if self.L is not None:
            return self.L
        K = self.get_K()
        # torch.potrf computes the Cholesky decomposition A s.t.
        # Sigma = AA^T and A is lower triangular.  
        self.L = torch.potrf(K, False)
        return self.L

    def get_f(self):
        if self.f is not None:
            return self.f
        L = self.get_L()
        g = self.g
        self.f = torch.mv(L, g)
        return self.f

    def log_likelihood(self):
        """
        Log Poisson likelihood p(y | f) (factorial term omitted).
        Note that f = log(Poisson mean).
        """
        Y = self.Y
        f = self.get_f()
        return (Y * f - f.exp()).sum()

    def log_prior(self):
        """
        Note: currently assumes a flat prior on the kernel parameters.
        The prior is just a standard normal on the whitened function values g.
        """
        g = self.g
        return -g.pow(2).sum()

    def neg_log_p(self):
        """
        Note: computes log likelihood only up to a constant additive factor
        """
        ll = self.log_likelihood()
        lp = self.log_prior()
        return -ll - lp

    def fit(self, num_steps=25, verbose=True, lr=0.0001):
        # LBFGS does not do a good job of fitting the kernel parameters.
        # We leave them fixed and optimize the latent function values.
        optim = torch.optim.LBFGS([self.g], lr=lr)
        clip = 500
        def closure():
            optim.zero_grad()
            self.clear()
            nlogp = self.neg_log_p()
            nlogp.backward()
            torch.nn.utils.clip_grad_norm([self.g], clip)
            return nlogp
        for i in range(num_steps):
            nlogp = closure()
            optim.step(closure)
            if verbose and i % 100 == 0:
                print("Iteration {} of {}".format(i, num_steps))
        self.clear()

    def predict(self, V):
        """
        Gets the predicted mean the GP for new inputs V.
        """
        V = Variable(V)
        # TODO
        pass

    def sample(self, v, num_samples=1, use_noise=False):
        """
        This is a placeholder - always returns the best-fit
        function values at U (or Poisson samples from them)
        """
        U = self.U.data
        ind = np.where(U == v)[0][0]

        mean = self.get_f().exp().data.numpy()[ind]
        if use_noise:
            return np.random.poisson(mean, size=num_samples)
        return mean * np.ones(num_samples)
        # TODO: real implementation
