import numpy as np

def relu_forward(x: np.ndarray) -> np.ndarray:
    """
    Computes ReLU(x) = max(0, x).
    """
    # Your code here
    # np.maximum(0, x) can also be used
    return np.where(x>0,x,0)
    pass