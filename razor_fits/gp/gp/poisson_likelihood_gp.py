import numpy as np
import torch
from torch.autograd import Variable

from .gaussian_process import GaussianProcess
from .kernel import make_log_par
from .hmc import run_hmc
from .annealing import run_annealing

def lsolve(A, B):
    """
    Computes the solution X to AX = B.
    """
    return torch.gesv(B, A)[0]

def log_det_from_chol(chol):
    """
    Gets the log determinant of a matrix given its Cholesky
    decomposition.
    """
    return 2 * chol.diag().log().sum()

def gauss_kl(mu1, chol1, mu2, chol2):
    """
    KL divergence KL(p1 || p2), where p1 and p2 are multivariate
    Gaussians with means mu1 and mu2, with chol1 and chol2
    the Cholesky decompositions of their covariance matrices.
    Formula taken from here:
    https://stats.stackexchange.com/questions/60680/
    kl-divergence-between-two-multivariate-gaussians/60699
    """
    # First term: log(det2/det1)
    term1 = log_det_from_chol(chol2) - log_det_from_chol(chol1)

    # Second term: -d
    term2 = -mu1.size(0)

    # Third term: trace(cov2^-1 cov1) -- compute by decomposing
    # cov2 and cov1 and using cyclic property:
    # trace(chol2^-1 chol1 * chol1^T chol2^T^-1)
    A = lsolve(chol2, chol1)
    term3 = torch.mm(A.t(), A).trace()

    # Fourth term: (mu2 - mu1)^T cov2^-1 (mu2 - mu1)
    B = lsolve(chol2, mu2 - mu1)
    term4 = B.dot(B)

    return 0.5 * (term1 + term2 + term3 + term4)


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

    def predict_mean(self, V):
        if self.samples is None:
            raise RuntimeError("Please sample before predicting")
        U = self.U
        L = self.get_L()
        mu_V = self.mean(V)
        K_UV = self.kernel.forward(U, V)
        A = self.lsolve(L, K_UV)

        # multiply by Cholesky to get log(Poisson mean) for each sample
        f = torch.matmul(L, 
                Variable(torch.Tensor(self.samples)).unsqueeze(-1)).squeeze()
        means = []
        for i in range(f.size(0)):
            # get the predicted mean from each available sample
            d = f[i] - self.mean(U)
            alpha = self.lsolve(L, d)

            # the second term is K_VU K^-1 d
            # which equals (L^-1 K_UV)^T (L^-1 d)
            means.append((mu_V + torch.mm(A.t(), alpha))[:, 0])
        return means

    def predict_K(self, V):
        U = self.U
        L = self.get_L()

        K_VV = self.kernel.forward(V, V)
        K_UV = self.kernel.forward(U, V)
        A = self.lsolve(L, K_UV)

        # the second term is K_VU K^-1 K_UV
        # which equals (L^-1 K_UV)^T (L^-1 K_UV)
        return K_VV - torch.mm(A.t(), A)

    def predict(self, V):
        """
        Gets the predicted mean and covariance of the GP for new inputs V.
        """
        V = Variable(V)
        pred_var = np.asarray([self.predict_K(
            Variable(
                torch.Tensor([float(v)]))).data.numpy() for v in V]).flatten()
        pred_means = self.predict_mean(V)
        # smear predicted means by predicted standard deviations
        preds = []
        for sample in pred_means:
            deltas = np.random.normal(size=pred_var.shape[0]) * pred_var
            preds.append(np.exp(sample.data.numpy() + deltas))
        return np.asarray(preds)

    def do_hmc(self, num_samples, epsilon=None, L_max=None, 
            print_every=200, verbose=False, extra_pars=[],
            abort_on_error=False):
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
                print_every=print_every, verbose=verbose,
                abort_on_error=abort_on_error)
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
            print_every=None, abort_on_error=False):
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
                    abort_on_error=abort_on_error,
                    print_every=print_every, verbose=verbose)
            self.samples = np.concatenate((self.samples, samples))
        elif self.samples is None:
            self.samples = self.do_hmc(num_samples, 
                    abort_on_error=abort_on_error,
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
        signal: signal strength relative to theoretical cross section
        num_true_signal: number of signal events actually injected
    """

    def __init__(self, kernel, U, Y, S, mean=None,
            hmc_epsilon=0.0001, hmc_L_max=10, num_true_signal=None):
        super(PoissonGPWithSignal, self).__init__(kernel, U, Y, mean,
                hmc_epsilon, hmc_L_max)

        self.S = Variable(S)
        self.signal = torch.nn.Parameter(torch.Tensor([5.0]))
        self.num_true_signal = num_true_signal

    def log_likelihood(self, include_signal=True):
        Y = self.Y
        mean = self.get_f().exp()
        if include_signal:
            s = self.signal * self.S
            mean = mean + s
            mean = torch.max(mean, Variable(torch.Tensor([1e-9]))) # force > 0
        return (Y * mean.log() - mean).sum()

    def fit(self, num_steps=1000, verbose=False, lr=0.0001,
            parameters=None):
        if parameters is None:
            parameters = [self.g, self.signal]
        super(PoissonGPWithSignal, self).fit(num_steps, verbose=verbose,
                lr=lr, parameters=parameters)

    def preds_from_samples(self, use_noise=False, include_signal=False):
        mean_samples = super(PoissonGPWithSignal, self).preds_from_samples(
                use_noise=False)
        if include_signal:
            sig_shape = self.S.data.numpy()
            ss = np.asarray([s[0] * sig_shape 
                for s in self.signal_samples])
            mean_samples = np.maximum(mean_samples + ss, 1e-9)
        if use_noise:
            return self.add_noise(mean_samples)
        return mean_samples

    def sample(self, num_samples=1, use_noise=False, verbose=False,
            abort_on_error=False):
        """
        This samples the observed data locations using HMC.
        Use predict() to get predictions for other input locations.
        """
        print_every = 100
        if num_samples > 2000:
            print_every = 2000
        self.samples, self.signal_samples = self.do_hmc(num_samples, 
                extra_pars=[self.signal], abort_on_error=abort_on_error,
                print_every=print_every, verbose=verbose)

        return self.preds_from_samples(use_noise=use_noise)


class AnnealingPoissonGP(PoissonGPWithSignal):
    """
    Poisson likelihood GP with inference performed with
    Annealed Importance Sampling to estimate the marginal likelihood.
    We take the signal strength to be fixed and compute the Bayes factor
    for the signal + background hypothesis (ignoring prior model probabilities)
    with respect to the background-only hypothesis.
    """

    def __init__(self, kernel, U, Y, S, mean=None, mu=1.0, 
            hmc_par_scheduler=None, num_true_signal=None):
        super(AnnealingPoissonGP, self).__init__(kernel, U, Y, S, mean,
                num_true_signal=num_true_signal)

        # fix the signal at the specified value
        self.signal = torch.nn.Parameter(torch.Tensor([mu]), 
                requires_grad=False)
        # this instructs the sampler how to adjust the HMC parameters
        self.hmc_par_scheduler = hmc_par_scheduler
        # this will hold sample importance weights
        self.log_importances = []
        # this will keep track of the number of samples per annealing run
        self.sample_multiplier = None

    def neg_log_p(self, beta=1.0):
        """
        This function has an extra parameter beta that is used 
        in the annealing algorithm.  beta = 1 corresponds to the 
        true log likelihood, and beta = 0 corresponds to the prior.
        """
        ll = self.log_likelihood()
        lp = self.log_prior()
        return -Variable(torch.Tensor([beta])) * ll - lp

    def preds_from_samples(self, use_noise=False, include_signal=False):
        # create dummy signal strength samples
        self.signal_samples = [self.signal.data.numpy() 
                for _ in range(self.samples.shape[0])]
        samples =  super(AnnealingPoissonGP, self).preds_from_samples(
                use_noise=use_noise, include_signal=include_signal)
        return samples, self.log_importances

    def sample_prior(self):
        # The prior is a standard normal for the parameters g.
        return [Variable(torch.randn(self.g.size(0)))]

    def sample(self, num_runs=1, num_hmc_samples=1, num_beta=100,
            verbose=False, abort_on_error=False, print_every=1,
            last_beta=None):
        """
        Runs the annealing procedure num_runs times.  Each run
        begins with a sample from the prior and ends with a sample
        from the posterior and its importance weight.  
        If num_hmc_samples > 1, the sampled point is used as the 
        beginning point for a standard HMC run of num_hmc_samples steps.
        If last_beta is provided, annealing will terminate at intermediate
        temperature last_beta (used for tuning the algorithm).
        """
        self.annealer = run_annealing(self, [self.g], num_runs, num_hmc_samples,
                num_beta, par_scheduler=self.hmc_par_scheduler,
                verbose=verbose, print_every=print_every, 
                abort_on_error=abort_on_error, last_beta=last_beta)
        # we don't have to throw any samples away as warm-up with this algo
        self.samples = np.asarray([s[0] for s in self.annealer.samples])
        self.log_importances = np.asarray(self.annealer.log_importances)
        self.sample_multiplier = num_hmc_samples
        return self.preds_from_samples()

    def get_marginal_likelihood(self):
        """
        Note: returns estimated marginal likelihood only up to a multiplicative
        constant (useful for computing Bayes factors)
        by averaging the importance weights.
        """
        # TODO
        pass


class VariationalPoissonGP(GaussianProcess):
    """
    GP with a Poisson likelihood, using a variational approximation
    to model the posterior.
    Attributes:
        m: mean of variational posterior
        S: covariance of variational posterior
        Z: inducing point locations (defaults to U)
        train_Z: bool, True if the inducing point locations should be optimized
    """

    def __init__(self, kernel, U, Y, mean=None, Z=None, train_Z=True):
        super(VariationalPoissonGP, self).__init__(kernel, U, Y, mean)
        if Z is None:
            Z = U.clone()
            default_Z = True
        else:
            default_Z = False
        self.log_Z = torch.nn.Parameter(Z.log(), requires_grad=train_Z)
        self.train_Z = train_Z
        self.clear()

        if default_Z:
            initial_m = (Y + 0.001).log().clone()
            initial_S_chol_ = self.get_L().clone().data / 10000
        else:
            initial_m = torch.ones(Z.size(0))
            initial_S_chol_ = torch.eye(Z.size(0)) / 1000000

        self.m = torch.nn.Parameter(initial_m)
        # We learn the posterior covariance matrix S by
        # learning its Cholesky decomposition, which is lower
        # triangular.  We start with a generic matrix (self.S_chol_) and 
        # then use tril() to force it lower triangular (self.S_chol).
        self.S_chol_ = torch.nn.Parameter(initial_S_chol_)

        self.clear()

    def clear(self):
        self.Z = None
        self.S_chol = None
        self.raw_K = None
        self.epsilon = None
        self.K = None
        self.L = None
        self.mu = None

    def get_Z(self):
        if self.Z is not None:
            return self.Z
        self.Z = self.log_Z.exp()
        return self.Z

    def get_S_chol(self):
        if self.S_chol is not None:
            return self.S_chol
        self.S_chol = torch.tril(self.S_chol_)
        return self.S_chol

    def get_raw_K(self):
        if self.raw_K is not None:
            return self.raw_K
        Z = self.get_Z()
        self.raw_K = self.kernel.forward(Z, Z)
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
        Z = self.get_Z()
        K = self.get_raw_K()
        epsilon = self.get_epsilon()
        noise = epsilon * Variable(torch.eye(Z.size(0)))
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

    def get_mu(self):
        if self.mu is not None:
            return self.mu
        self.mu = self.mean(self.get_Z())
        return self.mu

    def predict_mean(self, X):
        Z = self.get_Z()
        L = self.get_L()
        d = self.m - self.get_mu()

        mu_X = self.mean(X)
        K_ZX = self.kernel.forward(Z, X)
        A = self.lsolve(L, K_ZX)
        alpha = self.lsolve(L, d)

        return (mu_X + torch.mm(A.t(), alpha))[:, 0]

    def predict_cov(self, X):
        S_chol = self.get_S_chol()
        Z = self.get_Z()
        L = self.get_L()

        K_XX = self.kernel.forward(X, X)
        K_ZX = self.kernel.forward(Z, X)
        A = self.lsolve(L, K_ZX)

        term1 = K_XX
        term2 = torch.mm(A.t(), A)

        # The third term is K_XZ K^-1^T S K^-1 K_ZX
        # which we decompose in such a way as to not
        # evaluate any covariance matrix inverses directly
        B = self.lsolve(L, S_chol)
        C = torch.mm(A.t(), B)
        term3 = torch.mm(C, C.t())

        return term1 - term2 + term3

    def predict(self, v):
        v = Variable(v)
        pred_mean = self.predict_mean(v)
        pred_var = self.predict_cov(v)
        return pred_mean, pred_var

    def sample(self, v, num_samples=1):
        v = torch.Tensor([v])
        pred_mean, pred_sigma = self.predict(v)
        gauss = Variable(torch.randn(num_samples))
        return (gauss.data.numpy() * pred_sigma.sqrt().data.numpy() 
                + pred_mean.data.numpy())[0]

    def variational_expectation(self):
        """
        Computes the expectation value of log p(y|f)
        under the variational distribution q(f|m,S),
        summing over all observed data points.
        In the Poisson likelihood case, this can be computed 
        analytically as (y mu) - exp^(mu + sigma^2/2).
        """
        mean = self.predict_mean(self.U)
        var = self.predict_cov(self.U).diag()
        Y = self.Y
        return (Y * mean - (mean + var/2).exp()).sum()

    def variational_kl(self):
        """
        Computes the KL divergence between the variational
        posterior and the GP prior at the inducing locations.
        """
        return gauss_kl(self.m, self.get_S_chol(), self.get_mu(), self.get_L())

    def neg_elbo(self):
        """
        The variational evidence lower bound (ELBO) to optimize.
        """
        return -self.variational_expectation() + self.variational_kl()

    def fit(self, num_steps=1000, lr=0.01, verbose=False):
        to_train = [self.m, self.S_chol_]
        if self.train_Z:
            to_train.append(self.log_Z)
        optim = torch.optim.LBFGS(to_train, lr=lr)
        clip = 10
        Z_clip = 1
        def closure():
            optim.zero_grad()
            self.clear()
            neg_elbo = self.neg_elbo()
            neg_elbo.backward()
            torch.nn.utils.clip_grad_norm([self.m, self.S_chol_], clip)
            torch.nn.utils.clip_grad_norm([self.log_Z], Z_clip)
            return neg_elbo
        for i in range(num_steps):
            closure()
            optim.step(closure) 
            if verbose and i % 100 == 0:
                print("Iteration {}: -ELBO = {:.3f}".format(
                    i, self.neg_elbo().data.numpy()[0]))
        self.clear()

    def fit_adam(self, num_steps=10000, lr=0.01, verbose=False):
        to_train = [self.m, self.S_chol_]
        if self.train_Z:
            to_train.append(self.log_Z)
        optim = torch.optim.Adam(to_train, lr=lr)
        def closure():
            optim.zero_grad()
            self.clear()
            neg_elbo = self.neg_elbo()
            neg_elbo.backward()
            return neg_elbo
        for i in range(num_steps):
            closure()
            optim.step() 
            if verbose and i % 100 == 0:
                print("Iteration {}: -ELBO = {:.3f}".format(
                    i, self.neg_elbo().data.numpy()[0]))
        self.clear()

