import numpy as np

def kaiming_init(shape: tuple) -> np.ndarray:
    """
    Kaiming/He normal initialization.
    """
    # Your code here
    (fan_in,fan_out)=shape
    std=np.sqrt(2.0/fan_in).astype(np.float64)
    return np.random.normal(loc=0.0,scale=std,size=shape)
    pass