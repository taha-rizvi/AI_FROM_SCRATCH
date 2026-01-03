import numpy as np

def batch_matmul(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Computes Q @ K.T for batched multi-head attention.
    
    Args:
        Q: (B, H, S, D)
        K: (B, H, S, D)
        
    Returns:
        Result (B, H, S, S)
    """
    # Your code here
    attention=np.einsum("bhid,bhjd->bhij",Q,K,optimize=True)
    return attention
    pass