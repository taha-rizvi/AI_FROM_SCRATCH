import numpy as np
def kaiming_init(shape:tuple)->np.ndarray:
    fan_in,fan_out=shape
    std=np.sqrt(2.0/fan_in).astype(np.float64)
    return np.random.normal(loc=0.0,scale=std,size=shape)

class Linear:
    def __init__(self,input_size,output_size):
        self.weight=kaiming_init((input_size,output_size))
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
    