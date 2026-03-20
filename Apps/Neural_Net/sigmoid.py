import numpy as np
class sigmoid:
    def __init__(self):
        pass
    def __call__(self,x):

        self.out = np.where(
        x >= 0, 
        1 / (1 + np.exp(-x)), 
        np.exp(x) / (1 + np.exp(x))
    )
        return self.out
    def backward(self,dout):
        return self.out * (1 - self.out) * dout


    