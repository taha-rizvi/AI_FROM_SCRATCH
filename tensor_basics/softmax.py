import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Computes numerically stable softmax along the last axis.
    
    Args:
        x: Input logits (N, C)
        
    Returns:
        Probabilities (N, C)
    """
    # Your code here
    return np.exp(x-np.max(x))/(np.sum(np.exp(x-np.max(x)),axis=1)).reshape(-1,1)
    pass