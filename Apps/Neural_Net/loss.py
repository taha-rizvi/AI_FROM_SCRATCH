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

