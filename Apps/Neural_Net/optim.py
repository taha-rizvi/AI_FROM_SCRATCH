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
        

