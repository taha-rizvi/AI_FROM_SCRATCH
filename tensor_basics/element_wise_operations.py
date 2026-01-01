import numpy as np
from typing import Dict

def elementwise_ops(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes element-wise add, mul, and safe div.
    
    Args:
        a: First tensor
        b: Second tensor (same shape)
        
    Returns:
        Dictionary with keys "add", "mul", "div"
    """
    epsilon = 1e-8
    # Your code here
    ans={"add":np.array([]),
        "mul":np.array([]),
        "div":np.array([])}
    addition=np.add(a,b)
    mult=np.multiply(a,b)
    div=np.divide(a,b+epsilon)
    ans["add"],ans["mul"],ans["div"]=addition,mult,div
    return ans   
    pass