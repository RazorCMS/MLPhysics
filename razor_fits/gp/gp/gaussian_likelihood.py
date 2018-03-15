import numpy as np
import torch
from torch.autograd import Variable

from .gaussian_process import GaussianProcess

class GaussianLikelihoodGP(GaussianProcess):
    """
    GP with a Gaussian conditional likelihood.
    The calculations of the mean, the kernel matrix with and without noise,
    and the inverse of the kernel matrix with noise, are memoized
    in order to speed up prediction.  Clear the memoized values with clear()
    before e.g. evaluating the GP with new kernel hyperparameters.
    Attributes:
        sigma2: torch Tensor of the same shape as Y representing uncorrelated
                Gaussian noise on the outputs.  Here we are using the Gaussian
                likelihood as an approximation to the Poisson and take 
                sigma^2 = y pointwise (as in the Frate et al. paper).
        K: placeholder for the covariance matrix K.
        mu: placeholder for the prior mean.
        Sigma: placeholder for the covariance matrix Sigma = K + sigma^2 I.
        Sigma_inv: placeholder for the inverse of (K + sigma^2 I).
    """

    def __init__(self, kernel, U, Y, mean=None):
        super(GaussianLikelihoodGP, self).__init__(kernel, U, Y, mean)
        self.sigma2 = self.Y
        self.mu = None
        self.K = None
        self.Sigma = None
        self.Sigma_inv = None

    def clear(self):
        self.mu = None
        self.K = None
        self.Sigma = None
        self.Sigma_inv = None

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
        self.Sigma = self.get_K() + self.sigma2.diag()
        return self.Sigma

    def get_Sigma_inv(self):
        if self.Sigma_inv is not None:
            return self.Sigma_inv
        self.Sigma_inv = self.get_Sigma().inverse()
        return self.Sigma_inv

    def log_p(self):
        Y = self.Y
        B = self.B
        mu = self.get_mu()
        Sigma = self.get_Sigma()
        Sigma_inv = self.get_Sigma_inv()

        term1 = 0.5 * torch.det(Sigma).log()
        term2 = (Y - mu).transpose(0, 1).matmul(Sigma_inv).matmul(Y - mu)
        term3 = B / 2 * np.log(2 * np.pi)

        return -term1 - term2 - term3

    def predict_mean(self, V):
        U = self.U
        Y = self.Y
        mu = self.get_mu()
        Sigma_inv = self.get_Sigma_inv()

        mu_V = self.mean(V)
        K_VU = self.kernel.forward(V, U)

        return mu_V + K_VU.matmul(Sigma_inv).matmul(Y - mu)

    def predict_Sigma(self, V):
        U = self.U
        Sigma_inv = self.get_Sigma_inv()

        # Note: because we take the sigma^2 noise term from the observed data,
        # we do not have access to it when making predictions for new data.
        # To be correct one should add a diagonal noise matrix to Sigma_VV here.
        Sigma_VV = self.kernel.forward(V, V) 
        K_VU = self.kernel.forward(V, U)
        K_UV = K_VU.transpose(0, 1)

        return Sigma_VV - K_VU.matmul(Sigma_inv).matmul(K_UV)

    def predict(self, V):
        V = Variable(V)
        pred_mean = self.predict_mean(V)
        pred_Sigma = self.predict_Sigma(V) 
        return pred_mean, pred_Sigma
