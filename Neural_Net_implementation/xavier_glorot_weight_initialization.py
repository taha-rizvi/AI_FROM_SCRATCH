import numpy as np

def xavier_init(shape: tuple) -> np.ndarray:
    """
    Xavier/Glorot uniform initialization.
    """
    # Your code here
    (fan_in,fan_out)=shape
    a=np.sqrt(6/(fan_in+fan_out))
    return np.random.uniform(low=-a,high=a,size=(fan_in,fan_out))
    pass