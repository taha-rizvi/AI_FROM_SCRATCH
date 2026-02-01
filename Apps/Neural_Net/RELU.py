import numpy as np
class RELU:
    def __init__(self):
        
        pass
    def __call__(self,X):
        return max(0,X) #or np.where(x>0,x,0)
    
