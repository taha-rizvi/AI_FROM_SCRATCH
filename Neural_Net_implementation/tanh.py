import numpy as np
from typing import Dict

def tanh_ops(x: np.ndarray, dout: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes tanh forward and backward.
    """
    # Your code here
    ans={"out":np.array([]),
         "dx" :np.array([])}
    #we may add numerical stability like sigmoid function later
    out=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    ans["out"]=out
    ans["dx"]=dout*(1-out**2)
    return ans
    pass