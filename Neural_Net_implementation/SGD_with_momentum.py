import numpy as np
from typing import Tuple

def momentum_step(
    w: np.ndarray,
    dw: np.ndarray,
    v: np.ndarray,
    lr: float,
    momentum: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs SGD with momentum.
    Returns (w_new, v_new).
    """
    # Your code here
    v_new = momentum * v - lr * dw
    w_new = np.add(w, v_new)

    return (w_new, v_new)

    pass