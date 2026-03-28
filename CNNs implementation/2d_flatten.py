import numpy as np
def flatten(x):
    """
    x: np.ndarray of shape (N, C, H, W), dtype=np.float32
    returns: np.ndarray of shape (N, C*H*W), dtype=np.float32
    """
    N,C,H,W = x.shape
    return x.reshape((N,C*H*W)).astype(np.float32)
    raise NotImplementedError