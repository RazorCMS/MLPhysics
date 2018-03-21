import torch
from torch.autograd import Variable

class GaussianProcess(torch.nn.Module):
    """
    Base class for GP models.
    Derived classes should implement the sample() and 
    neg_log_p (negative log marginal likelihood) functions.
    Attributes:
        kernel: Kernel object defining the similarity measure
        U (torch Tensor): locations of observed data points
        Y (torch Tensor): observed data values
        B (int): number of data points
        mean: function mapping inputs to prior means
    """

    def __init__(self, kernel, U, Y, mean=None):
        super(GaussianProcess, self).__init__()
        self.kernel = kernel
        self.U = Variable(U)
        self.Y = Variable(Y)
        self.B = Y.size()[0]
        if mean is None:
            mean = torch.zeros_like
        self.mean = mean

    def neg_log_p(self):
        """Returns the negative log marginal likelihood of the observed data."""
        raise NotImplementedError(
                "Please do not use the GaussianProcess class directly")

    def sample(self, v, num_samples=1, use_noise=False, verbose=False):
        """
        Samples the function value(s) at a new input location v.
        Returns a numpy array of samples.
        If use_noise is False, should return samples of the function value.
        If it's True, should return data samples generated from the function
        (so randomness on both the function and the observation 
            process is included).
        """
        raise NotImplementedError(
                "Please do not use the GaussianProcess class directly")
