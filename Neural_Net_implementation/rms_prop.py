import numpy as np
from typing import Tuple

def rmsprop_step(w: np.ndarray, dw: np.ndarray, cache: np.ndarray, lr: float, decay: float = 0.99, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    RMSProp update. Returns (w_new, cache_new).
    """
    # Your code here
    cache_new=decay*cache +(1-decay)*(dw**2)
    w_new=w-lr*dw/(np.sqrt(cache_new)+eps)
    return (w_new,cache_new)
    pass