import numpy as np
from typing import Dict
def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float | np.ndarray]:
    """
    Computes MSE loss and gradient.
    """
    # Your code here
    ans={}
    ans["loss"]=np.mean(np.mean((y_pred-y_true)**2,axis=1)).astype(np.float64) #can also do a complicated version where we can first mean over axis and then mean again over other axis.
    ans["dx"]=(2*(np.mean(np.mean(y_pred-y_true,axis=1)))).astype(np.float64)
    return ans
    pass