import numpy as np

def cross_entropy_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes mean cross-entropy loss.
    
    Args:
        probs: (N, C) probabilities
        targets: (N,) integer class indices
        
    Returns:
        Scalar float loss
    """
    # Your code here
    epsilon=1e-9
    p=probs[np.arange(len(probs)),targets]
    p=np.add(p,epsilon)
    return -np.mean(np.log(p)).astype(np.float64)
    pass