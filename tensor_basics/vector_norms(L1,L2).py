import numpy as np
from typing import Dict

def compute_norms(x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes L1 and L2 norms for a batch of vectors.
    
    Args:
        x: Input matrix of shape (N, D)
        
    Returns:
        Dictionary with keys "l1" and "l2", each containing an array of shape (N,)
    """
    # Your code here
    ans={"l1":np.array([]),
         "l2":np.array([])}
    ans["l1"]=np.linalg.norm(x,ord=1,axis=1)
    ans["l2"]=np.linalg.norm(x,axis=1)
    return ans
    pass

