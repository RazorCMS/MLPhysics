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
                Gaussian noise on the outputs.  
                If not provided as an input to __init__, it defaults to
                the choice in the Frate et al. paper, sigma^2 = y pointwise.
        K: placeholder for the covariance matrix K.
        mu: placeholder for the prior mean.
        Sigma: placeholder for the covariance matrix Sigma = K + sigma^2 I.
        Sigma_inv: placeholder for the inverse of (K + sigma^2 I).
    """

    def __init__(self, kernel, U, Y, mean=None, sigma2=None):
        super(GaussianLikelihoodGP, self).__init__(kernel, U, Y, mean)
        if sigma2 is None:
            self.sigma2 = self.Y
        else:
            self.sigma2 = torch.nn.Parameter(torch.Tensor([sigma2]))
        self.mu = None
        self.K = None
        self.Sigma = None
        self.Sigma_inv = None
        self.log_det_Sigma = None

    def clear(self):
        self.mu = None
        self.K = None
        self.Sigma = None
        self.Sigma_inv = None
        self.log_det_Sigma = None

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
        if self.sigma2.size()[0] > 1: # vector of pointwise variances
            noise = self.sigma2.diag()
        else: # scalar variance
            noise = self.sigma2 * Variable(torch.eye(self.U.size(0)))
        self.Sigma = self.get_K() + noise
        return self.Sigma

    def get_Sigma_inv(self):
        if self.Sigma_inv is not None:
            return self.Sigma_inv
        self.Sigma_inv = self.get_Sigma().inverse()
        return self.Sigma_inv

    def get_log_det_Sigma(self):
        if self.log_det_Sigma is not None:
            return self.log_det_Sigma
        Sigma = self.get_Sigma()
        # torch.potrf computes the Cholesky decomposition A s.t.
        # Sigma = AA^T and A is lower triangular.  The determinant
        # of A is sqrt(det Sigma), so log det A is 0.5 log det Sigma.
        # The determinant is A is the product of its diagonal entries.
        chol = torch.potrf(Sigma, False)
        self.log_det_Sigma = 2 * chol.diag().log().sum()
        return self.log_det_Sigma

    def neg_log_p(self):
        Y = self.Y
        B = self.B
        mu = self.get_mu()
        Sigma = self.get_Sigma()
        Sigma_inv = self.get_Sigma_inv()

        term1 = 0.5 * self.get_log_det_Sigma()
        term2 = 0.5 * (Y - mu).dot(torch.mv(Sigma_inv, Y - mu))
        term3 = 0.5 * B * np.log(2 * np.pi)

        return term1 + term2 + term3

    def predict_mean(self, V):
        U = self.U
        Y = self.Y
        mu = self.get_mu()
        Sigma_inv = self.get_Sigma_inv()

        mu_V = self.mean(V)
        K_VU = self.kernel.forward(V, U)

        return mu_V + torch.mv(K_VU, torch.mv(Sigma_inv, Y - mu))

    def predict_Sigma(self, V):
        U = self.U
        Sigma_inv = self.get_Sigma_inv()

        # Note: because we take the sigma^2 noise term from the observed data,
        # we do not have access to it when making predictions for new data.
        # To be correct one should add a diagonal noise matrix to Sigma_VV here.
        Sigma_VV = self.kernel.forward(V, V) 
        K_VU = self.kernel.forward(V, U)
        K_UV = K_VU.transpose(0, 1)

        out = Sigma_VV - torch.mm(K_VU, torch.mm(Sigma_inv, K_UV))
        # Add a tiny multiple of the identity to avoid 
        # the matrix being considered non-positive-semidefinite by numpy
        epsilon = 1e-8
        min_eig = torch.min(torch.eig(out)[0][:,0]).data.numpy()[0]
        if min_eig < 0:
            print("Min eigenvalue is negative: {}".format(min_eig))
            epsilon = max(epsilon, -10 * min_eig)
        out = out + Variable(torch.Tensor([epsilon])) * Variable(torch.eye(out.size()[0]))
        return out

    def predict(self, V):
        """
        Gets the predicted mean and covariance of the GP for new inputs V.
        """
        V = Variable(V)
        pred_mean = self.predict_mean(V)
        pred_Sigma = self.predict_Sigma(V) 
        return pred_mean, pred_Sigma

    def sample(self, v, num_samples=1):
        v = torch.Tensor([v])
        pred_mean, pred_Sigma = self.predict(v)
        gauss = Variable(torch.randn(num_samples))
        return (gauss.data.numpy() * pred_Sigma.data.numpy() 
                + pred_mean.data.numpy())[0]
