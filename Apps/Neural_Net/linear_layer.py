import numpy as np
class Linear:
    def __init__(self,input_size,output_size):
        self.weight=np.random.randn(input_size,output_size)*0.01
        self.bias=np.zeros((1,output_size))
    def __call__(self,X):
        return np.matmul(X,self.weight)+self.bias
    def parameters(self):
        return [self.weight,self.bias]
    