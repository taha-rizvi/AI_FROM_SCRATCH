import numpy as np

class CrossEntropy:
    def __init__(self,out,y):
        self.out=out
        self.y=y
    def item(self):
        epsilon=1e-9
        p=self.out[np.arange(len(self.out)),self.y]
        p=np.add(p,epsilon)
        return -np.mean(np.log(p))
    def backward(self):
        y_i=np.full(self.out.shape,0)
        y_i[np.arange(len(self.y)),self.y]=1

        gradient=np.subtract(self.out,y_i)
        dout=gradient/len(self.out)
        return dout

class binaryCrossEntropy:
    def __init__(self,out,y):
        self.out=out
        self.y=y
    def item(self):
        eps=1e-9
        return -np.mean(self.y*np.log(self.out+eps)+(1-self.y)*np.log(1-self.out+eps)).astype(np.float64)
    
    def backward(self):
        eps=1e-9
        dout = (
        (self.out - self.y) /
        ((self.out + eps) * (1 - self.out + eps))
    ) / len(self.y)

        return dout

class CategoricalCrossEntropy:
    def __init__(self,out,y):
        self.y=y
        self.out=out
    def item(self):
        
        return -np.mean(np.sum(self.y * np.log(self.out), axis=1))
    def backward(self):   
    
        return -self.y / (self.out * len(self.y))
    
class MSE:
    def __init__(self,out,y):
        self.y=y
        self.out=out
    def item(self):

        return np.mean((self.out - self.y)**2)
    def backward(self):   
        n=self.out.size
        return (2 / n) * (self.out - self.y)
    