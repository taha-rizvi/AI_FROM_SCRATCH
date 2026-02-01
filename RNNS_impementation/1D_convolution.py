import numpy as np
def conv1d_forward(x, kernel, bias=None, stride=1, padding=0, dilation=1):
    """
    x: np.ndarray of shape (N, C_in, L), dtype=np.float32 - input sequence
    kernel: np.ndarray of shape (C_out, C_in, k), dtype=np.float32 - convolution kernel
    bias: None or np.ndarray of shape (C_out,), dtype=np.float32 - bias vector
    stride: int - stride along temporal dimension
    padding: int - padding on both sides
    dilation: int - dilation rate (default 1)
    returns: np.ndarray of shape (N, C_out, L_out), dtype=np.float32
    """
    
    N,C_in,L_in=x.shape
   
    C_out,C_in,k=kernel.shape
    if(dilation>1):
        keff=(k-1)*dilation+1
    else:
        keff=k
    L_out=(L_in+2*padding-keff)//(stride)+1
    L_padded=L_in+2*padding
    y=np.zeros((N,C_out,L_out))
    for n in range(N):
        for c in range(C_out):
            for i in range(L_out):

                start =i*stride+padding
                accumulator=0

                for channel in range(C_in):
                    for j in range(k):
                        input_index=i*stride+j-padding
                        accumulator+=np.dot(x[n][channel][input_index],kernel[c][channel][j])
                accumulator+=bias[c]  
                y[n][c][i]=accumulator

    return y

    raise NotImplementedError