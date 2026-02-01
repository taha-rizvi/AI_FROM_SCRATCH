import numpy as np
class softmax:
    def __init__(self):

        pass
    def __call__(self,x):
        return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)),axis=1, keepdims=True)