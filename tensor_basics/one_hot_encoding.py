import numpy as np

def one_hot_encode(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Encodes integer labels into one-hot vectors.
    
    Args:
        indices: 1D array of class indices (N,)
        num_classes: Total number of classes C
        
    Returns:
        One-hot matrix of shape (N, C)
    """
    # Your code here
    ans=np.full((len(indices),num_classes),fill_value=0)
    ans[np.arange(len(indices)),indices]=1
    return ans.astype(np.float64)
    pass
