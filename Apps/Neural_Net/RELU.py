import numpy as np
class RELU:
    def __init__(self):
        
        pass
    def parameters(self):
        return []
    def __call__(self,x):
        return np.where(x>0,x,0)
    
