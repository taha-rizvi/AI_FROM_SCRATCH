import numpy as np

def categorical_ce_backward(probs, targets):
    """
    Computes gradient of Softmax + Categorical Cross Entropy
    """
    # Your code here
    indices = np.argmax(probs)
    
    y_i = np.full(probs.shape, 0)
    y_i[np.arange(len(targets)), targets] = 1

    gradient = np.subtract(probs, y_i)
    ans = gradient / len(probs)
    
    return ans.astype(np.float32)