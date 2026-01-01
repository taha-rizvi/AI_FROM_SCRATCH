import numpy as np
from typing import Dict, Union

def tensor_reductions(x: np.ndarray, axis: int) -> Dict[str, Union[np.ndarray, float]]:
    """
    Computes sum, mean, max, argmax along axis.
    """
    # Your code here
    ans={"sum":np.array([]),
        "mean":np.array([]),
        "max":np.array([]),
        "argmax":np.array([])}
    
    
    ans["sum"]=np.sum(x,axis=axis)
    ans["mean"]=np.mean(x,axis=axis)


    ans["max"]=np.max(x,axis=axis)

    ans["argmax"]=np.argmax(x,axis=axis)

    return ans


    pass
