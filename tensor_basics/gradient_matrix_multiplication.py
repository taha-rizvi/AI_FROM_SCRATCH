import numpy as np
from typing import Dict

def grad_matmul(grad_C: np.ndarray, A: np.ndarray, B: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes gradients for C = AB.
    
    Args:
        grad_C: Upstream gradient dL/dC (M, N)
        A: Input A (M, K)
        B: Input B (K, N)
        
    Returns:
        Dict with "grad_A" and "grad_B"
    """
    # Your code here
    ans={"grad_A":np.array([]),
         "grad_B":np.array([])}
    ans["grad_A"] = np.matmul(grad_C,B.T)
    ans["grad_B"] = np.matmul(A.T,grad_C)
    return ans
    pass