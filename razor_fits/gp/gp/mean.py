import torch
from torch.autograd import Variable

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

        self.N = torch.nn.Parameter(torch.Tensor([N]))
        self.slope = torch.nn.Parameter(torch.Tensor([slope]))

    def forward(self, U):
        return self.N * torch.exp(-self.slope * U)
