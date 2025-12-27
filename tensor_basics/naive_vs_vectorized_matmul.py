import numpy as np
#Naive implementation of matrix multiplication using 3 nested loops is often much slower than a vectorized implementation that leverages optimized linear algebra libraries.
def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes matrix product C = AB using 3 nested loops.
    """
    # Get dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    
    # Initialize result matrix with zeros
    C = np.zeros((M, N))
    
    # Your code here
    
    
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i][j]+=A[i][k]*B[k][j]
                    
    return C

def matmul_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes matrix product C = AB using vectorized operations.
    """
    # Your code here
    return np.dot(A,B)
    pass

