import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, lr=0.1, max_iters=100, tolerance=1e-6):
        self.lr = lr
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.costs = []
        self.trajectory_pos = []
    
    def objective_fn(self, params):
        x, y = params
        return x**2 + y**2 + 10*np.sin(x) + 10*np.cos(y)
    
    def gradient_fn(self, params):
        x, y = params
        dz_dx = 2*x + 10*np.cos(x)
        dz_dy = 2*y - 10*np.sin(y)
        return np.array([dz_dx, dz_dy])
    
    def step(self, initial_params):
        params = np.array(initial_params, dtype=float)
        self.costs = []
        self.trajectory_pos = []
        gradient_magnitude=[]
        
        for iter in range(self.max_iters):
            self.trajectory_pos.append(params.copy())
            self.costs.append(self.objective_fn(params))
            
            grad = self.gradient_fn(params)
            params = params - self.lr * grad
            grad_mag=np.sqrt(grad[0]**2+grad[1]**2)

            # if len(self.costs) > 1 and abs(self.objective_fn(params) - self.costs[-1]) < self.tolerance:
            if(grad_mag<self.tolerance):
                self.trajectory_pos.append(params.copy()) 
                gradient_magnitude.append(grad_mag) 
                self.costs.append(self.objective_fn(params))
                break
        
        return np.array(self.trajectory_pos), np.array(self.costs)



    



          


