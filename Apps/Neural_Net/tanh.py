import numpy as np
class Tanh:

    def __call__(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1 - self.out**2)