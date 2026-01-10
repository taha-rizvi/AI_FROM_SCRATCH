import numpy as np
from typing import Tuple, Optional

def dropout_forward(x: np.ndarray, p: float, train: bool = True, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverted dropout forward. Returns (out, mask).
    """
    # Your code here
    if seed is not None:
        np.random.seed(seed)

    if train:
        # Create a mask of the same shape as x
        # Elements are 1 with probability (1-p) and 0 with probability p
        mask=np.random.binomial(n=1,p=(1-p),size=x.shape)
        
        # Scale the output by 1/(1-p) to maintain expected value
        out = (x * mask) / (1 - p)
        return out, mask
    else:
        return x, None
    pass