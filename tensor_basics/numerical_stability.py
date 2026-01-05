import numpy as np

def log_sum_exp(x: np.ndarray) -> np.ndarray:
    """
    Computes log(sum(exp(x))) stably along the last axis.
    
    Args:
        x: Input (N, D)
        
    Returns:
        Result (N,)
    """
    # Your code here
    return np.add(np.max(x),np.log(np.sum(np.exp(x-np.max(x)),axis=1)))
    pass