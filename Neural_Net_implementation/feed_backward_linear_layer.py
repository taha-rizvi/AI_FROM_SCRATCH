import numpy as np
from typing import Dict

def linear_backward(dout: np.ndarray, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes dx, dw, db for y = x @ w + b.
    
    Args:
        dout: Upstream gradient (N, Dout)
        x: Input (N, Din)
        w: Weights (Din, Dout)
        b: Bias (Dout,)
        
    Returns:
        Dict with "dx", "dw", "db"
    """
    # Your code here
    ans={"dx":np.array([]),
         "dw":np.array([]),
         "db":np.array([])}
    ans["dx"]=np.matmul(dout,w.T)
    ans["dw"]=np.matmul(x.T,dout)
    ans["db"]=np.sum(dout,axis=0)
    return ans
    pass
