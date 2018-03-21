import numpy as np
import torch
from torch.autograd import Variable

def make_log_par(par):
    """
    Input: float or numpy array
    Output: torch Parameter representing the log of the input.
    """
    # check if par is a scalar and wrap it in a list if not
    try:
        iter(par)
    except TypeError:
        par = [par]
    return torch.nn.Parameter(torch.Tensor(np.log(par)))


class Kernel(torch.nn.Module):
    """
    Base class for GP kernels.
    Derived classes should implement the forward() function that
    specifies how the kernel operates on two input vectors.
    """

    def __init__(self):
        super(Kernel, self).__init__()

    def expand_inputs(self, U, V):
        """
        Converts inputs U and V into matrices suitable for 
        many kernel operations.  
        U becomes a matrix whose ith row is repeated copies of 
            the ith element of U.
        V becomes a matrix whose jth column is repeated copies of
            the jth element of V.
        """
        u_len = U.size()[0]
        v_len = V.size()[0]
        input_dims = self.get_input_dims(U)
        mat1 = U.unsqueeze(1).expand(u_len, v_len, *input_dims)
        mat2 = V.unsqueeze(0).expand(u_len, v_len, *input_dims)
        return mat1, mat2

    def get_input_dims(self, U):
        """
        Returns the dimensionality of the data space (i.e.
        the dimension of each entry of U) as a torch Size object.
        """
        return U.size()[1:]

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
        assert ell > 0, "ell must be positive"
        assert alpha > 0, "alpha must be positive"
        super(SquaredExponentialKernel, self).__init__()

        self.log_ell = make_log_par(ell)
        self.log_alpha = make_log_par(alpha)

    def forward(self, U, V):
        assert (len(U.size()) < 3 and len(V.size()) < 3), (
                "This kernel is not implemented for more than two dimensions")
        mat1, mat2 = self.expand_inputs(U, V)
        diff = mat1 - mat2
        ell = torch.exp(self.log_ell)
        alpha = torch.exp(self.log_alpha)
        input_dims = U.size()[1:]
        if len(input_dims):
            ell = ell.expand(u_len, v_len, *input_dims)
            norm = diff.pow(2).sum(dim=-1) / (2 * ell * ell)
        else:
            norm = diff.pow(2) / (2 * ell * ell)
        return alpha * torch.exp(-norm)


class ConstantKernel(Kernel):
    """
    Kernel giving constant correlation strength A for all inputs.
    """
    
    def __init__(self, A):
        assert A > 0, "A must be positive"
        super(ConstantKernel, self).__init__()

        self.log_A = make_log_par(A)

    def forward(self, U, V):
        return torch.exp(self.log_A)


class ExponentialDecayKernel(Kernel):
    """
    Kernel whose strength decreases exponentially with the values of its inputs
    (note: NOT with the distance between them): K(u, v) = exp(-a(u + v)).
    The offset parameter represents the lower bound of the fit range
    and is not optimized during fitting.
    """
    
    def __init__(self, a, offset=650):
        assert a > 0, "a must be positive"
        super(ExponentialDecayKernel, self).__init__()

        self.log_a = make_log_par(a)
        self.offset = Variable(torch.Tensor([offset]))

    def forward(self, U, V):
        # TODO: handle multidimensional a
        mat1, mat2 = self.expand_inputs(U, V)
        a = torch.exp(self.log_a)
        return torch.exp(-a * (mat1 + mat2 - 2 * self.offset))


class GibbsKernel(Kernel):
    """
    Squared exponential kernel with position dependent covariance length.
    See http://www.gaussianprocess.org/gpml/chapters/RW4.pdf.
    The covariance length l(x) can be an arbitrary positive function of x
    (to be defined in subclasses).
    """
    
    def __init__(self, alpha):
        super(GibbsKernel, self).__init__()

        self.log_alpha = make_log_par(alpha)

    def l(self, x):
        raise NotImplementedError(
                "Please do not use the GibbsKernel class directly")

    def forward(self, U, V):
        mat1, mat2 = self.expand_inputs(U, V)
        l1 = self.l(mat1)
        l2 = self.l(mat2)
        lsquared = l1.pow(2) + l2.pow(2)
        alpha = self.log_alpha.exp()

        # This prefactor is needed to make the kernel
        # positive definite for all possible inputs
        prefactor = alpha * torch.sqrt(2 * l1 * l2 / lsquared)

        return prefactor * torch.exp(-(mat1 - mat2).pow(2) / lsquared)


class LinearGibbsKernel(GibbsKernel):
    """
    Squared exponential kernel with linearly changing correlation strength,
    l(x) = bx + c.
    """

    def __init__(self, alpha, b, c):
        super(LinearGibbsKernel, self).__init__(alpha)

        self.log_b = make_log_par(b)
        self.log_c = make_log_par(c)

    def l(self, x):
        # TODO: handle multidimensional b, c
        b = self.log_b.exp()
        c = self.log_c.exp()
        return b * x + c


class PhysicsInspiredKernel(Kernel):
    """
    Product of two kernels:
        1) an exponential kernel (parameter a)
        2) a linear Gibbs kernel (parameters A, b, c)
    """

    def __init__(self, A, a, b, c):
        super(PhysicsInspiredKernel, self).__init__()

        self.exp_kernel = ExponentialDecayKernel(a)
        self.linear_kernel = LinearGibbsKernel(A, b, c)

    @property
    def log_a(self):return self.exp_kernel.log_a
    
    @property
    def log_b(self):return self.linear_kernel.log_b
    
    @property
    def log_c(self):return self.linear_kernel.log_c
    
    @property
    def log_A(self): return self.linear_kernel.log_alpha

    def forward(self, U, V):
        exp = self.exp_kernel.forward(U, V)
        linear = self.linear_kernel.forward(U, V)
        return exp * linear
