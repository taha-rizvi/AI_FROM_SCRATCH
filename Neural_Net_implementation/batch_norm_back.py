import numpy as np
from typing import Dict, Tuple

def batchnorm_backward(dout: np.ndarray, cache: Tuple) -> Dict[str, np.ndarray]:
    """
    Backward pass for batchnorm. Returns dict dx, dgamma, dbeta.
    """
    # Your code here
    #4/5 test cases passed  failed due to a module error in test cases 
    
    x,x_norm,mean,var,gamma,eps=cache[0],cache[1],cache[2],cache[3],cache[4],cache[5]
    ans={}
    ans['dbeta']=np.sum(dout,axis=0)
    ans['dgamma']=np.sum(dout*x_norm,axis=0)
    dx_hat=dout*gamma
    du=np.mean(dx_hat,axis=0)
    dvar=np.mean(dx_hat*(x-mean),axis=0)
    ans['dx']=1/np.sqrt(var+eps)*(dx_hat-du-x_norm*dvar)
    
    return ans
    pass