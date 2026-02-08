import numpy as np
class Linear:
    def __init__(self,input_size,output_size):
        self.weight=np.random.randn(input_size,output_size)*0.01
        self.bias=np.zeros((output_size))
    def __call__(self,X):
        self.X=X
        return np.matmul(X,self.weight)+self.bias
    def backward(self,dout):
        self.dw=np.matmul(self.X.T,dout)
        self.db=np.sum(dout,axis=0)
        dx=np.matmul(dout,self.weight.T)
        return dx

    def parameters(self):
        return [self.weight,self.bias]
    