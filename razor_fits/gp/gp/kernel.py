import torch
from torch.autograd import Variable

class Kernel(object):
    """
    Base class for GP kernels.
    Derived classes should implement the forward() function that
    specifies how the kernel operates on two input vectors.
    Attributes:
        parameters: list of torch Variables representing hyperparameters
    """

    def __init__(self):
        self.parameters = []

    def forward(self, U, V):
        """
        Evaluates the kernel function on two vector inputs
        (which should be torch Tensors)
        of length u and v, returning a (u x v) matrix.
        """
        raise NotImplementedError("Please do not use the Kernel class directly")


class SquaredExponentialKernel(Kernel):
    """
    Squared exponential or RBF kernel.
    Attributes:
        ell (float, list, or numpy array): characteristic correlation length(s)
        alpha (positive float): 
    """

    def __init__(self, ell, alpha):
        assert alpha > 0, "alpha must be positive"

        # check if ell is a scalar and wrap it in a list if not
        try:
            iter(ell)
        except TypeError:
            ell = [ell]
        self.ell = Variable(torch.Tensor(ell), requires_grad=True)
        self.alpha = Variable(torch.Tensor([alpha]), requires_grad=True)
        self.parameters = [self.ell, self.alpha]

    def forward(self, U, V):
        assert (len(U.size()) < 3 and len(V.size()) < 3), (
                "This kernel is not implemented for more than two dimensions")
        u_len = U.size()[0]
        v_len = V.size()[0]
        other_dims = U.size()[1:]
        mat1 = U.unsqueeze(1).expand(u_len, v_len, *other_dims)
        mat2 = V.unsqueeze(0).expand(u_len, v_len, *other_dims)
        # Now mat1's row i is repeated copies of the ith element of U
        # and mat2's column j is repeated copies of the jth element of V
        diff = mat1 - mat2
        if len(other_dims):
            ell = self.ell.expand(u_len, v_len, *other_dims)
            norm = diff.pow(2).sum(dim=-1) / (2 * ell * ell)
        else:
            norm = diff.pow(2) / (2 * self.ell * self.ell)
        return self.alpha * torch.exp(norm)
