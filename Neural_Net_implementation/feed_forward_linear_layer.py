import numpy as np

def linear_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes y = x @ w + b
    
    Args:
        x: (N, Din)
        w: (Din, Dout)
        b: (Dout,)
        
    Returns:
        y: (N, Dout)
    """
    # Your code here
    return np.add(np.matmul(x,w),b)
    pass