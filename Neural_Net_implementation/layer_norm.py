import numpy as np
from typing import Tuple, Dict

def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, Tuple]:
    """
    LayerNorm forward. Returns (out, cache).
    """
    # Your code here
    mean_layer=np.mean(x,axis=1,keepdims=True)
    var_layer=np.mean((x-mean_layer)**2,axis=1,keepdims=True)
    
    x_norm=(x-mean_layer)/np.sqrt(var_layer+eps)
    out=gamma*x_norm + beta
    cache={'x':x,'x_norm':x_norm,'mean':mean_layer,'var':var_layer,'gamma':gamma,'eps':eps}
    return (out,cache)
    pass

def layernorm_backward(dout: np.ndarray, cache: Tuple) -> Dict[str, np.ndarray]:
    """
    LayerNorm backward. Returns dict dx, dgamma, dbeta.
    """
    # Your code here
    x,x_norm,mean,var,gamma,eps=cache['x'],cache['x_norm'],cache['mean'],cache['var'],cache['gamma'],cache['eps']
    dbeta=np.sum(dout,axis=0)
    dgamma=np.sum(dout*x_norm,axis=0)
    dx_hat=dout*gamma
    du=np.mean(dx_hat,axis=1,keepdims=True)
    dvar=np.mean(dx_hat*(x-mean),axis=1,keepdims=True)
    dx=1/(np.sqrt(var+eps))*(dx_hat-du-x_norm*dvar)
    ans={"dbeta":dbeta,
         "dgamma":dgamma,
         "dx":dx
        }
    return ans
    pass