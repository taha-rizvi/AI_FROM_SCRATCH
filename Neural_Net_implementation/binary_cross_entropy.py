import numpy as np
from typing import Dict

def bce_loss(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float | np.ndarray]:
    """
    Computes BCE loss and gradient.
    """
    # Your code here
    eps=1e-9
    
    ans={}
    n=len(y_pred)
    ans["loss"]=-np.mean(y_true*np.log(y_pred+eps)+(1-y_true)*np.log(1-y_pred+eps)).astype(np.float64)
    ans["dx"]=((y_pred-y_true)/n).astype(np.float64)
    return ans
    pass