import numpy as np
def im2col(x, kernel_size, stride=1, padding=0):
    """
    Convert image patches to columns for efficient convolution.
    
    x: np.ndarray of shape (N, C, H_in, W_in), dtype=np.float32
    kernel_size: int or tuple of two ints (K_h, K_w)
    stride: int or tuple of two ints (s_h, s_w)
    padding: int or tuple of two ints (p_h, p_w)
    returns: np.ndarray of shape (C * K_h * K_w, N * H_out * W_out), dtype=np.float32
    """
    #WARNING : PLEASE GO THROUGH THE THEORY BEHIND THIS OPERATION.
    N,C,H_in,W_in=x.shape
    if isinstance(kernel_size,tuple):
        K_h,K_w=kernel_size
    else:
        K_h=K_w=kernel_size
    if stride is None:
        s_h, s_w = K_h, K_w
    elif isinstance(stride, tuple):
        s_h, s_w = stride
    else:
        s_h = s_w = stride
    if isinstance(padding,tuple):
        p_h,p_w=padding
    else:
        p_h=p_w=padding
    H_out=(H_in+2*p_h-K_h)//s_h+1
    W_out=(W_in+2*p_w-K_w)//s_w+1
    x_padded=np.pad(x,((0,0),(0,0),(p_h,p_h),(p_w,p_w)),mode='constant',constant_values=-np.inf)
    col_matrix=np.zeros((C*K_h*K_w,N*H_out*W_out),dtype=np.float32)
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                start_h=i*s_h
                start_w=j*s_w
                end_h=start_h+K_h
                end_w=start_w+K_w
                patch=x_padded[n,:,start_h:end_h,start_w:end_w]  #try to understand from here(you didn't got the column major and row major thing)
                flat_patch = patch.transpose(0, 2, 1).reshape(-1)
                col_ind=n*H_out*W_out+i*W_out+j
                col_matrix[:,col_ind]=flat_patch
    return col_matrix

    raise NotImplementedError