import torch
from torch.autograd import Variable

from .kernel import make_log_par

class Mean(torch.nn.Module):
    """
    Base class for GP mean functions.
    """

    def __init__(self):
        super(Mean, self).__init__()

    def forward(self, U):
        """
        Evaluates the kernel function on a vector input.
        """
        raise NotImplementedError("Please do not use the Mean class directly")


class ExponentialMean(Mean):
    """
    Exponential mean function.
    Attributes:
        N: value at zero
        slope: rate of exponential decrease
    """

    def __init__(self, N, slope):
        super(ExponentialMean, self).__init__()
        assert N > 0, "N must be positive"
        assert slope > 0, "slope must be positive"

        self.log_N = make_log_par(N)
        self.log_slope = make_log_par(slope)

    def forward(self, U):
        N = self.log_N.exp()
        slope = self.log_slope.exp()
        return N * torch.exp(-slope * U)
