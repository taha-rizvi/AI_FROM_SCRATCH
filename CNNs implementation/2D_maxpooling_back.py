import numpy as np
def max_pool_backward(dout, x, kernel_size, stride=None, padding=0):
    """
    dout: np.ndarray of shape (N, C, H_out, W_out), dtype=np.float32
    x: np.ndarray of shape (N, C, H_in, W_in), dtype=np.float32
    kernel_size: int or tuple of two ints (K_h, K_w)
    stride: None, int, or tuple of two ints (s_h, s_w). If None, defaults to kernel_size.
    padding: int or tuple of two ints (p_h, p_w)
    returns: np.ndarray of shape (N, C, H_in, W_in), dtype=np.float32
    """
    N,C,H_out,W_out=dout.shape
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
    x_padded=np.pad(x,((0,0),(0,0),(p_h,p_h),(p_w,p_w)),mode='constant',constant_values=-np.inf)
    dx_padded=np.zeros_like(x_padded)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    start_h=i*s_h
                    start_w=j*s_w
                    max_val=float('-inf')
                    grad=dout[n,c,i,j]
                    max_h,max_w=-1,-1
                    # for kh in range(K_h):
                    #     for kw in range(K_w):
                    #         h=start_h+kh
                    #         w=start_w+kw
                    #         if x_padded[n,c,h,w]>max_val:
                    #             max_val=x_padded[n,c,h,w]
                    #             max_h,max_w=h,w
                    # dx_padded[n,c,max_h,max_w]+=grad
                    window = x_padded[n, c, start_h:start_h+K_h, start_w:start_w+K_w]

                    max_val = np.max(window)

                    mask = (window == max_val)
                    num_max = np.sum(mask)

                    dx_padded[n, c, start_h:start_h+K_h, start_w:start_w+K_w] += (mask * grad) / num_max      
    dx=dx_padded[:,:,p_h:p_h+H_in,p_w:p_w+W_in]
    return dx
    raise NotImplementedError