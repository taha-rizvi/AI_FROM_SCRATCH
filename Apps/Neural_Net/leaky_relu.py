import numpy as np
class LeakyReLU:

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        dx = np.ones_like(self.x)
        dx[self.x < 0] = self.alpha
        return dout * dx