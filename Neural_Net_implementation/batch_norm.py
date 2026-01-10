import numpy as np
from typing import Tuple, Optional

def batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5, momentum: float = 0.9, running_mean: Optional[np.ndarray] = None, running_var: Optional[np.ndarray] = None, train: bool = True):
    """
    BatchNorm forward. Returns (out, cache, running_mean_new, running_var_new)
    """
    # Your code here
    if(train):
        batch_mean=np.mean(x,axis=0)
        batch_var=np.mean((x-batch_mean)**2,axis=0)
        x_norm=(x-batch_mean)/np.sqrt(batch_var+eps)
        out=gamma*x_norm+beta
        cache={'x':x,'x_norm':x_norm,'mean':batch_mean,'var':batch_var,'gamma':gamma,'eps':eps}
        running_mean_new=momentum*running_mean+(1-momentum)*batch_mean
        running_var_new=momentum*running_var+(1-momentum)*batch_var
        return out,cache,running_mean_new,running_var_new
    else:
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        return out, None, running_mean, running_var

    pass