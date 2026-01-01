import numpy as np

def broadcast_ops(X: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Computes (X + b) * w using broadcasting.
    
    Args:
        X: Input matrix of shape (N, D)
        b: Bias vector of shape (D,)
        w: Weight vector of shape (N,)
        
    Returns:
        Resulting matrix of shape (N, D)
    """
    # Your code here
    #we adding X into b automatically broadcasts b to (N) from (D,)
    #so no need to broadcast it explicitly
    #w needs to be broadcasted from (N,) to (N,1)
    w=w.reshape(-1,1)
    return (X + b) * w
    #notes: the first argument in reshape is the row number and second one is column but a wildcard like -1 is used when you want numpy to automatically determine that dimension based on the other given dimensions.