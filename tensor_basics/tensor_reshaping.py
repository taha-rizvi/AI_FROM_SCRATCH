import numpy as np

def reshape_and_transpose(x: np.ndarray, B: int, C: int, H: int, W: int) -> np.ndarray:
    """
    Reshapes flat x to (B, C, H, W) then transposes to (B, H, W, C).
    """
    # Your code here
    bchw=x.reshape((B,C,H,W))
    return bchw.transpose(0,2,3,1)
    pass