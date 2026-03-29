import numpy as np
class SGD:
    def __init__(self,params,lr):
        self.params=params
        self.lr=lr
    def step(self):
        for layer in self.params:
            layer.weight-=self.lr*layer.dw
            layer.bias-=self.lr*layer.db
    def zero_grad(self):
        for layer in self.params:
            layer.dw*=0
            layer.db*=0
        
class ADAM:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.t = 0  
        
        
        self.m_w = [np.zeros_like(layer.weight) for layer in params]
        self.v_w = [np.zeros_like(layer.weight) for layer in params]
        
        self.m_b = [np.zeros_like(layer.bias) for layer in params]
        self.v_b = [np.zeros_like(layer.bias) for layer in params]

    def step(self):
        self.t += 1
        
        for i, layer in enumerate(self.params):
            dw = layer.dw
            db = layer.db
            
            
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dw
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dw ** 2)
            
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            
            layer.weight -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            
            
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db ** 2)
            
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)

    def zero_grad(self):
        
        for layer in self.params:
            layer.dw*=0
            layer.db*=0
class Momentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        
        self.v_w = [np.zeros_like(layer.weight) for layer in params]
        self.v_b = [np.zeros_like(layer.bias) for layer in params]

    def step(self):
        for i, layer in enumerate(self.params):
            self.v_w[i] = self.momentum * self.v_w[i] + self.lr * layer.dw
            self.v_b[i] = self.momentum * self.v_b[i] + self.lr * layer.db
            
            layer.weight -= self.v_w[i]
            layer.bias   -= self.v_b[i]

    def zero_grad(self):
        for layer in self.params:
            layer.dw *= 0
            layer.db *= 0


class RMSprop:
    def __init__(self, params, lr=0.001, decay=0.9, eps=1e-8):
        self.params = params
        self.lr = lr
        self.decay = decay
        self.eps = eps
        
        self.v_w = [np.zeros_like(layer.weight) for layer in params]
        self.v_b = [np.zeros_like(layer.bias) for layer in params]

    def step(self):
        for i, layer in enumerate(self.params):
            self.v_w[i] = self.decay * self.v_w[i] + (1 - self.decay) * (layer.dw ** 2)
            self.v_b[i] = self.decay * self.v_b[i] + (1 - self.decay) * (layer.db ** 2)
            
            layer.weight -= self.lr * layer.dw / (np.sqrt(self.v_w[i]) + self.eps)
            layer.bias   -= self.lr * layer.db / (np.sqrt(self.v_b[i]) + self.eps)

    def zero_grad(self):
        for layer in self.params:
            layer.dw *= 0
            layer.db *= 0


class Adagrad:
    def __init__(self, params, lr=0.01, eps=1e-8):
        self.params = params
        self.lr = lr
        self.eps = eps
        
        # Accumulates squared gradients forever (never resets)
        self.G_w = [np.zeros_like(layer.weight) for layer in params]
        self.G_b = [np.zeros_like(layer.bias) for layer in params]

    def step(self):
        for i, layer in enumerate(self.params):
            self.G_w[i] += layer.dw ** 2
            self.G_b[i] += layer.db ** 2
            
            layer.weight -= self.lr * layer.dw / (np.sqrt(self.G_w[i]) + self.eps)
            layer.bias   -= self.lr * layer.db / (np.sqrt(self.G_b[i]) + self.eps)

    def zero_grad(self):
        for layer in self.params:
            layer.dw *= 0
            layer.db *= 0