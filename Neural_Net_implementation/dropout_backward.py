import numpy as np
from typing import Optional

def dropout_backward(dout: np.ndarray, mask: Optional[np.ndarray], p: float, train: bool = True) -> np.ndarray:
    """
    Backward pass for inverted dropout.
    """
    # Your code here
    if(train):
        return mask/(1-p)*dout
    else:
        return dout
    pass