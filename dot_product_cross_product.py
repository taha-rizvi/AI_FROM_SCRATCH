import numpy as np
from typing import Dict

def vector_products(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes dot and cross products for batches of 3D vectors.
    
    Args:
        a: Shape (N, 3)
        b: Shape (N, 3)
        
    Returns:
        Dict with "dot" (N,) and "cross" (N, 3)
    """
    # Your code here
    ans={"dot":np.array([]),
         "cross":np.array([])}
    ans["dot"]=np.sum(a*b,axis=1)
    ax,ay,az=a[:,0],a[:,1],a[:,2]
    bx,by,bz=b[:,0],b[:,1],b[:,2]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    ans["cross"]=np.array((cx,cy,cz)).T #again transpose to get correct shape
    return ans
    pass

