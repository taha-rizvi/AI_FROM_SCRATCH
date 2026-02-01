import numpy as np
class sigmoid:
    def __init__(self):
        pass
    def __call__(self,X):
        return 1/(1+np.exp(-X))
    