import numpy as np
def conv2d_forward(x, kernel, bias=None, stride=1, padding=0):
    """
    x: np.ndarray of shape (N, C_in, H_in, W_in), dtype=np.float32
    kernel: np.ndarray of shape (C_out, C_in, K_h, K_w), dtype=np.float32
    bias: None or np.ndarray of shape (C_out,), dtype=np.float32
    stride: int or tuple of two ints (s_h, s_w)
    padding: int or tuple of two ints (p_h, p_w)
    returns: np.ndarray of shape (N, C_out, H_out, W_out), dtype=np.float32
    """
    N,C_in,H_in,W_in=x.shape
    C_out,C_in,K_h,K_w=kernel.shape
    if isinstance(padding,tuple):
        p_h,p_w=padding
    else:
        
        p_h=p_w=padding
    if isinstance(stride,tuple):
        s_h,s_w=stride
        
    else:
        
        s_h=s_w=stride

    H_out=int((H_in+2*p_h-K_h)/s_h+1)
    W_out=int((W_in+2*p_w-K_w)/s_w+1)
    x_padded = np.pad(
        x,
        ((0,0), (0,0), (p_h,p_h), (p_w,p_w)),  ##in the formula they subtracted like they added stride but u actually pad using numpy 
        mode='constant'
    )
    y=np.zeros((N,C_out,H_out,W_out),dtype=np.float32)
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):      #for the base 4 loops just see the the shape of y
                if isinstance(stride,tuple):
                    start_h=i*s_h
                else:
                    start_h=i*stride
                
                for j in range(W_out):
                    if isinstance(stride,tuple):
                        start_w=j*s_w
                    else:
                        start_w=j*stride
                    
                    accumulator=0.0
                    for ci in range(C_in):   #for the next 4 loops just see the right hand side shapes of x and kernel
                        
                        for kh in range(K_h):
                            h_in=start_h+kh    #u wrongly accumulated it as start_h+=kh
                            for kw in range(K_w):
                                w_in=start_w+kw
                                accumulator+=x_padded[n,ci,h_in,w_in]*kernel[co,ci,kh,kw]
                    if bias is not None:
                        accumulator+=bias[co]        
                    y[n,co,i,j]=accumulator  
    
    return y





    raise NotImplementedError