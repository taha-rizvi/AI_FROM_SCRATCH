def time_distributed_dense(x, W, b):
    """
    x: np.ndarray of shape (N, T, d_in), dtype=np.float32 - input sequence
    W: np.ndarray of shape (d_out, d_in), dtype=np.float32 - weight matrix
    b: np.ndarray of shape (d_out,), dtype=np.float32 - bias vector
    returns: np.ndarray of shape (N, T, d_out), dtype=np.float32 - output sequence
    """

    N,T,din=x.shape
    dout,_=W.shape
    x_flat=x.reshape(N*T,-1)
    y_flat=x_flat@W.T+b
    return y_flat.reshape(N,T,-1)
    raise NotImplementedError