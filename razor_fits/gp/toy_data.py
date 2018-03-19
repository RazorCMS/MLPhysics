import numpy as np
import torch

def gen_exponential_toy(N, slope, sigma2, num_bins, mr_min, mr_max):
    """
    Generate a random realization from an exponential mean function
    with Gaussian noise.  Returns a dictionary of torch Tensors of the 
    same form as the data generators in razor_data.py.
    """
    bin_width = (mr_max - mr_min) / num_bins
    bin_centers = np.arange(mr_min, mr_max, bin_width) + bin_width / 2
    mean = N * np.exp(-slope * bin_centers)
    smeared = np.random.normal(mean, np.sqrt(sigma2))
    return {'u':torch.Tensor(bin_centers), 
            'y':torch.Tensor(smeared)}
