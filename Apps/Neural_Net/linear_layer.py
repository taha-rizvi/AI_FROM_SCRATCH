import numpy as np
class Linear:
    def __init__(self,output_size,input_size):
        self.w=np.random.randn(input_size,output_size)*0.01
        self.b=np.zeros((1,output_size))
    def __call__(self,X):
        return np.matmul(X,self.w)+self.b
    