import numpy as np
class RELU:
    def __init__(self):
        
        pass
    def parameters(self):
        return []
    def __call__(self,x):
        self.x=x
        return np.where(x>0,x,0)
    def backward(self,dout):
        mask=self.x>0
        return dout*mask
    
