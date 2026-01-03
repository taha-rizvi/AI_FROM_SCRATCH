import numpy as np
from typing import Tuple

def grad_sum(grad_y: float, x_shape: Tuple[int]) -> np.ndarray:
    """
    Computes gradient of x given gradient of sum(x).
    
    Args:
        grad_y: Scalar gradient dL/dy
        x_shape: Shape of the input tensor x
        
    Returns:
        Gradient dL/dx of shape x_shape
    """
    # Your code here
    
    return np.full(x_shape,grad_y)
    pass