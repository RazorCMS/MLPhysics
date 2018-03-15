import torch
from torch.autograd import Variable

class GaussianProcess(object):
    """
    Base class for GP models.
    Derived classes should implement the predict() and 
    logp() functions.
    Attributes:
        kernel: Kernel object defining the similarity measure
        U (torch Tensor): locations of observed data points
        Y (torch Tensor): observed data values
        B (int): number of data points
        mean: function mapping inputs to prior means
    """

    def __init__(self, kernel, U, Y, mean=None):
        self.kernel = kernel
        self.U = Variable(U)
        self.Y = Variable(Y)
        self.B = Y.size()[0]
        if mean is None:
            mean = torch.zeros_like
        self.mean = mean

    def log_p(self):
        """Returns the log marginal likelihood of the observed data."""
        raise NotImplementedError(
                "Please do not use the GaussianProcess class directly")

    def predict(self, V):
        """
        Predicts the function value(s) at new input locations V.
        The return type may be different for different kinds of GP.
        """
        raise NotImplementedError(
                "Please do not use the GaussianProcess class directly")
