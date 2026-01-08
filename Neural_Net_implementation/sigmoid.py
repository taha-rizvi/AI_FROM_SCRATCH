import numpy as np
from typing import Dict

def sigmoid_ops(x: np.ndarray, dout: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes sigmoid forward and backward.
    """
    # Your code here
    out = np.where(
        x >= 0, 
        1 / (1 + np.exp(-x)), 
        np.exp(x) / (1 + np.exp(x))
    )
    
    # Backward Pass: (out * (1 - out)) * dout
    dx = out * (1 - out) * dout
    
    return {"out": out, "dx": dx}
    pass