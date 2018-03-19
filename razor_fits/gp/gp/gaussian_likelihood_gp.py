import numpy as np
import torch
from torch.autograd import Variable

from .gaussian_process import GaussianProcess
from .kernel import make_log_par

class GaussianLikelihoodGP(GaussianProcess):
    """
    GP with a Gaussian conditional likelihood.
    The calculations of the mean, the kernel matrix with and without noise,
    and the Cholesky decomposition of the kernel matrix,
    are memoized in order to speed up prediction.  
    Clear the memoized values with clear()
    before e.g. evaluating the GP with new kernel hyperparameters.
    Optimized using ideas from
    https://github.com/t-vi/candlegp/blob/master/candlegp/densities.py
    Attributes:
        sigma2: torch Tensor of the same shape as Y representing uncorrelated
                Gaussian noise on the outputs.  
                If not provided as an input to __init__, it defaults to
                the choice in the Frate et al. paper, sigma^2 = y pointwise.
        K: placeholder for the covariance matrix K.
        mu: placeholder for the prior mean.
        Sigma: placeholder for the covariance matrix Sigma = K + sigma^2 I.
        uses_poisson_noise: True if we use the square root of the observed data
            as the noise matrix (default if no sigma2 is provided)
    """

    def __init__(self, kernel, U, Y, mean=None, sigma2=None):
        super(GaussianLikelihoodGP, self).__init__(kernel, U, Y, mean)
        if sigma2 is None:
            # Take noise std as sqrt(observed data counts)
            self.uses_poisson_noise = True
            self.log_sigma2 = self.Y.log()
        else:
            self.uses_poisson_noise = False
            self.log_sigma2 = make_log_par(sigma2)
        self.mu = None
        self.K = None
        self.Sigma = None
        self.Sigma_chol = None
        self.log_det_Sigma = None

    def clear(self):
        self.mu = None
        self.K = None
        self.Sigma = None
        self.Sigma_chol = None
        self.log_det_Sigma = None

    def add_noise(self, cov, noise=None):
        """
        Input: covariance matrix without noise
        Output: covariance matrix with noise added
        """
        if noise is None:
            noise = self.log_sigma2.exp()

        if self.uses_poisson_noise or noise.size(0) > 1: 
            # vector of pointwise variances
            return cov + noise.diag()
        else: # scalar variance
            return cov + noise * Variable(torch.eye(cov.size(0)))

    def lsolve(self, A, B):
        """
        Computes the solution X to AX = B.
        """
        return torch.gesv(B, A)[0]

    def get_mu(self):
        if self.mu is not None:
            return self.mu
        self.mu = self.mean(self.U)
        return self.mu

    def get_K(self):
        if self.K is not None:
            return self.K
        self.K = self.kernel.forward(self.U, self.U)
        return self.K

    def get_Sigma(self):
        if self.Sigma is not None:
            return self.Sigma
        K = self.get_K()
        return self.add_noise(K)

    def get_Sigma_chol(self):
        if self.Sigma_chol is not None:
            return self.Sigma_chol
        Sigma = self.get_Sigma()
        # torch.potrf computes the Cholesky decomposition A s.t.
        # Sigma = AA^T and A is lower triangular.  
        self.Sigma_chol = torch.potrf(Sigma, False)
        return self.Sigma_chol

    def get_log_det_Sigma(self):
        if self.log_det_Sigma is not None:
            return self.log_det_Sigma
        chol = self.get_Sigma_chol()
        # The determinant of chol is sqrt(det Sigma), 
        # so log det chol is 0.5 log det Sigma.
        # The determinant of chol is the product of its diagonal entries
        # because chol is lower triangular.
        self.log_det_Sigma = 2 * chol.diag().log().sum()
        return self.log_det_Sigma

    def neg_log_p(self):
        Y = self.Y
        B = self.B
        d = Y - self.get_mu()
        chol = self.get_Sigma_chol()
        alpha = self.lsolve(chol, d)

        term1 = 0.5 * self.get_log_det_Sigma()
        # this term is 1/2 d^T Sigma^-1 d 
        # which equals 1/2 (chol^-1 d)^T (chol^-1 d)
        term2 = 0.5 * alpha.pow(2).sum()
        term3 = 0.5 * B * np.log(2 * np.pi)

        return term1 + term2 + term3

    def predict_mean(self, V):
        U = self.U
        Y = self.Y
        d = Y - self.get_mu()

        mu_V = self.mean(V)

        K_UV = self.kernel.forward(U, V)
        chol = self.get_Sigma_chol()
        A = self.lsolve(chol, K_UV)
        alpha = self.lsolve(chol, d) 

        # the second term is K_VU Sigma^-1 d
        # which equals (chol^-1 K_UV)^T (chol^-1 d)
        return (mu_V + torch.mm(A.t(), alpha))[:, 0]

    def predict_Sigma(self, V, use_noise=False, noise_vector=None):
        K_VV = self.kernel.forward(V, V) 
        if use_noise:
            if self.uses_poisson_noise:
                # Note: because we take the sigma^2 noise term from the 
                # observed data, we do not have access to it when making 
                # predictions for new data.  In this case a noise vector
                # must be supplied here.
                if noise_vector is None:
                    raise ValueError("Please supply a Poisson noise vector")
                Sigma_VV = self.add_noise(K_VV, noise=noise_vector)
            else:
                Sigma_VV = self.add_noise(K_VV)
        else:
            Sigma_VV = K_VV

        U = self.U
        K_UV = self.kernel.forward(U, V)
        chol = self.get_Sigma_chol()
        A = self.lsolve(chol, K_UV)

        # the second term is K_VU Sigma^-1 K_UV
        # which equals (chol^-1 K_UV)^T (chol^-1 K_UV)
        out = Sigma_VV - torch.mm(A.t(), A)

        # warn if there are negative eigenvalues
        # (indicates numerical instability)
        min_eig = torch.min(torch.eig(out)[0][:,0]).data.numpy()[0]
        if min_eig < 0:
            print("Warning: min eigenvalue is negative: {}".format(min_eig))
        return out

    def predict(self, V, use_noise=False, poisson_noise_vector=None):
        """
        Gets the predicted mean and covariance of the GP for new inputs V.
        A custom noise vector (for the Poisson noise case) can be provided.
        """
        V = Variable(V)
        if poisson_noise_vector is not None:
            poisson_noise_vector = Variable(poisson_noise_vector)
        pred_mean = self.predict_mean(V)
        pred_Sigma = self.predict_Sigma(V, use_noise=use_noise,
                noise_vector=poisson_noise_vector) 
        return pred_mean, pred_Sigma

    def sample(self, v, num_samples=1, use_noise=False, 
            poisson_noise_vector=None):
        v = torch.Tensor([v])
        pred_mean, pred_Sigma = self.predict(v, use_noise=use_noise,
                poisson_noise_vector=poisson_noise_vector)
        gauss = Variable(torch.randn(num_samples))
        return (gauss.data.numpy() * pred_Sigma.sqrt().data.numpy() 
                + pred_mean.data.numpy())[0]
