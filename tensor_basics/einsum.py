import numpy as np
from typing import Dict, Union

def einsum_ops(A: np.ndarray, B: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
    """
    Computes basic ops using np.einsum.
    
    Args:
        A: (N, D)
        B: (D, M)
        
    Returns:
        Dict with "transpose", "sum", "row_sum", "col_sum", "matmul"
    """
    # Your code here
    ans={"transpose":np.array([]),
         "sum":np.array([]),
         "row_sum":np.array([]),
         "col_sum":np.array([]),
         "matmul":np.array([])}
    ans["transpose"]=np.einsum("ij->ji",A)
    ans["sum"]=np.einsum("ij->",A)
    ans["row_sum"]=np.einsum("ij->i",A)
    ans["col_sum"]=np.einsum("ij->j",A)
    ans["matmul"]=np.einsum("ij,jk->ik",A.astype(np.float64),B.astype(np.float64))
    return ans
    pass