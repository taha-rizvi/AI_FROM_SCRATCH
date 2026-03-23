import numpy as np

def conv2d_backward(dout, x, kernel, bias=None, stride=1, padding=0):
    N, C_in, H_in, W_in = x.shape
    C_out, _, K_h, K_w = kernel.shape
    _, _, H_out, W_out = dout.shape

    # stride
    if isinstance(stride, tuple):
        s_h, s_w = stride
    else:
        s_h = s_w = stride

    # padding
    if isinstance(padding, tuple):
        p_h, p_w = padding
    else:
        p_h = p_w = padding

    # pad input
    x_padded = np.pad(
        x,
        ((0,0),(0,0),(p_h,p_h),(p_w,p_w)),
        mode='constant'
    )

    dx_padded = np.zeros_like(x_padded)
    dkernel = np.zeros_like(kernel)
    dbias = np.zeros((C_out,), dtype=np.float32) if bias is not None else None

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):

                    grad = dout[n, co, i, j]

                    start_h = i * s_h
                    start_w = j * s_w

                    for ci in range(C_in):
                        for kh in range(K_h):
                            for kw in range(K_w):

                                h = start_h + kh
                                w = start_w + kw

                                # dkernel
                                dkernel[co, ci, kh, kw] += (
                                    grad * x_padded[n, ci, h, w]
                                )

                                # dx
                                dx_padded[n, ci, h, w] += (
                                    grad * kernel[co, ci, kh, kw]
                                )

    # remove padding from dx
    dx = dx_padded[:, :, p_h:p_h+H_in, p_w:p_w+W_in]

    # dbias
    if bias is not None:
        for co in range(C_out):
            dbias[co] = np.sum(dout[:, co, :, :])

    return dx, dkernel, dbias