import numpy as np
def max_pool_forward(x, kernel_size, stride=None, padding=0):
    """
    x: np.ndarray of shape (N, C, H_in, W_in), dtype=np.float32
    kernel_size: int or tuple of two ints (K_h, K_w)
    stride: None, int, or tuple of two ints (s_h, s_w). If None, defaults to kernel_size.
    padding: int or tuple of two ints (p_h, p_w)
    returns: np.ndarray of shape (N, C, H_out, W_out), dtype=np.float32
    """
    N,C,H_in,W_in=x.shape
    if isinstance(kernel_size,tuple):
        K_h,K_w=kernel_size
    else:
        K_h=K_w=kernel_size
    if isinstance(stride,tuple):
        s_h,s_w=stride
    else:
        if stride is None:
            s_h=s_w=kernel_size
        else:
            s_h=s_w=stride
    if isinstance(padding,tuple):
        p_h,p_w=padding
    else:
        p_h=p_w=padding
    
    H_out=(H_in+2*p_h-K_h)//s_h+1
    W_out=(W_in+2*p_w-K_w)//s_w+1
    x_padded=np.pad(x,((0,0),(0,0),(p_h,p_h),(p_w,p_w)),mode='constant',constant_values=-np.inf)
    y=np.zeros((N,C,H_out,W_out),dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                start_h=i*s_h
                for j in range(W_out):
                    start_w=j*s_w
                    x_max=float('-inf')
                    for kh in range(K_h):
                        for kw in range(K_w):
                            h=start_h+kh
                            w=start_w+kw
                            x_max=max(x_padded[n,c,h,w],x_max)
                    y[n,c,i,j]=x_max
    return y

    raise NotImplementedError