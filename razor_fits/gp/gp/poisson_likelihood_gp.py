import numpy as np
import torch
from torch.autograd import Variable

from .gaussian_process import GaussianProcess
from .kernel import make_log_par
from .hmc import run_hmc

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
        samples: list of HMC samples at the observed data point locations
        hmc_epsilon, hmc_L_max: tuning parameters for Hamiltonian MC.
    """

    def __init__(self, kernel, U, Y, mean=None,
            hmc_epsilon=0.0001, hmc_L_max=10):
        super(PoissonLikelihoodGP, self).__init__(kernel, U, Y, mean)
        self.hmc_epsilon = hmc_epsilon
        self.hmc_L_max = hmc_L_max

        self.mu = None
        self.K = None
        self.epsilon = None
        self.raw_K = None
        self.L = None 
        self.f = None 
        self.samples = None

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
        return -0.5 * g.pow(2).sum()

    def neg_log_p(self):
        """
        Note: computes log likelihood only up to a constant additive factor
        Note: this is the conditional likelihood P(f | y), not the marginal 
        likelihood (which is intractable in the Poisson case).
        """
        ll = self.log_likelihood()
        lp = self.log_prior()
        return -ll - lp

    def fit(self, num_steps=1000, verbose=True, lr=0.0001,
            parameters=None):
        # LBFGS does not do a good job of fitting the kernel parameters.
        # We leave them fixed and optimize the latent function values.
        if parameters is None:
            parameters = [self.g]
        optim = torch.optim.LBFGS(parameters, lr=lr)
        clip = 500
        def closure():
            optim.zero_grad()
            self.clear()
            nlogp = self.neg_log_p()
            nlogp.backward()
            torch.nn.utils.clip_grad_norm(parameters, clip)
            return nlogp
        for i in range(num_steps):
            nlogp = closure()
            optim.step(closure)
            if verbose and i % 100 == 0:
                print("Iteration {} of {}".format(i, num_steps))
        self.clear()

    def predict(self, V):
        """
        Gets the predicted mean of the GP for new inputs V.
        """
        V = Variable(V)
        # TODO
        pass

    def do_hmc(self, num_samples, epsilon=None, L_max=None, 
            print_every=200, verbose=False, extra_pars=[]):
        """
        Runs HMC for num_samples steps with parameters epsilon, L_max.
        By default, only the whitened function values g are sampled.
        Additional parameters can be sampled if provided 
        as a list in extra_pars; samples of these .
        """
        if epsilon is None:
            epsilon = self.hmc_epsilon
        if L_max is None:
            L_max = self.hmc_L_max
        pars = [self.g] + extra_pars
        hmc = run_hmc(self, pars, num_samples=(num_samples*2),
                epsilon=epsilon, L_max=L_max, 
                print_every=print_every, verbose=verbose)
        # Values of g are first in the list of sampled parameters.
        # Take the second half of the sampled parameters
        # (discarding the first half as warm-up)
        g_samples = np.asarray([s[0] for s in hmc.samples[num_samples:]])
        if extra_pars:
            par_samples = [s[1:] for s in hmc.samples[num_samples:]]
            return g_samples, par_samples
        else:
            return g_samples

    def add_noise(self, f):
        """
        Sample data values from the function values.
        """
        return np.random.poisson(f)

    def preds_from_samples(self, use_noise=False):
        """
        Converts samples of g to samples of the latent Poisson mean
        or random draws from the Poisson mean (if use_noise is True)
        """
        if self.samples is None:
            raise ValueError("Need to sample first")
        L = self.get_L().data.numpy()
        mean_samples = np.exp(np.matmul(
            L, np.expand_dims(self.samples, -1))).squeeze()
        if use_noise:
            return self.add_noise(mean_samples)
        return mean_samples

    def sample(self, v=None, num_samples=1, use_noise=False, verbose=False,
            print_every=None):
        """
        Currently this samples the observed data locations using HMC.
        Predicting non-data locations is not supported yet.
        """
        if print_every is None:
            print_every = 100
            if num_samples > 2000:
                print_every = 2000
        # Generate more samples if needed
        if self.samples is not None and len(self.samples) < num_samples:
            samples_needed = num_samples - len(self.samples)
            samples = self.do_hmc(samples_needed, 
                    print_every=print_every, verbose=verbose)
            self.samples = np.concatenate((self.samples, samples))
        elif self.samples is None:
            self.samples = self.do_hmc(num_samples, 
                    print_every=print_every, verbose=verbose)
        samples = self.preds_from_samples(use_noise=use_noise)
        if v is not None:
            # Figure out which data point we are sampling
            U = self.U.data
            try:
                ind = np.where(U == v)[0][0]
            except IndexError:
                raise ValueError(
                        "Sampling location must be one of the input points")
            samples = samples[:, ind]
        return samples


class PoissonGPWithSignal(PoissonLikelihoodGP):
    """
    Poisson likelihood GP with a variable-strength signal included.
    Attributes: 
        S: expected signal counts at theoretical cross section
        log_signal: signal strength relative to theoretical cross section
    """

    def __init__(self, kernel, U, Y, S, mean=None,
            hmc_epsilon=0.0001, hmc_L_max=10):
        super(PoissonGPWithSignal, self).__init__(kernel, U, Y, mean,
                hmc_epsilon, hmc_L_max)

        self.S = Variable(S)
        self.log_signal = make_log_par(1.0)

    def log_likelihood(self):
        Y = self.Y
        b = self.get_f().exp()
        s = self.log_signal.exp() * self.S
        mean = s + b
        return (Y * mean.log() - mean).sum()

    def fit(self, num_steps=1000, verbose=False, lr=0.0001,
            parameters=None):
        if parameters is None:
            parameters = [self.g, self.log_signal]
        super(PoissonGPWithSignal, self).fit(num_steps, verbose=verbose,
                lr=lr, parameters=parameters)

    def preds_from_samples(self, use_noise=False, include_signal=False):
        mean_samples = super(PoissonGPWithSignal, self).preds_from_samples(
                use_noise=False)
        if include_signal:
            sig_shape = self.S.data.numpy()
            ss = np.asarray([np.exp(s[0]) * sig_shape 
                for s in self.log_signal_samples])
            mean_samples = mean_samples + ss
        if use_noise:
            return self.add_noise(mean_samples)
        return mean_samples

    def sample(self, num_samples=1, use_noise=False, verbose=False):
        """
        Currently this samples the observed data locations using HMC.
        Predicting non-data locations is not supported yet.
        """
        print_every = 100
        if num_samples > 2000:
            print_every = 2000
        self.samples, self.log_signal_samples = self.do_hmc(num_samples, 
                extra_pars=[self.log_signal],
                print_every=print_every, verbose=verbose)

        return self.preds_from_samples(use_noise=use_noise)
