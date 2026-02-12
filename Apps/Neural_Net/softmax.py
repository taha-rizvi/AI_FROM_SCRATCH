import numpy as np
class softmax:
    def __init__(self):

        pass
    def __call__(self,x):
        self.output= np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)),axis=1, keepdims=True)
        return self.output
    def backward(self,dout):
        return self.output*(dout-np.sum(dout*self.output,axis=1,keepdims=True))