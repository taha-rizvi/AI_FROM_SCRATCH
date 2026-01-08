import numpy as np

def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Computes dx for ReLU.
    """
    # Your code here
    mask=x>0
    return dout*mask
    pass