

import numpy as np
def char_rnn_forward(x_chars, char_to_idx, embedding_weight, W_x, W_h, b, W_o, b_o):
    """
    x_chars: list of str or np.ndarray of ints - input character sequence
    char_to_idx: dict mapping chars to ints, or None if x_chars is already ints
    embedding_weight: np.ndarray of shape (V, d_e), dtype=np.float32 - character embeddings
    W_x: np.ndarray of shape (d_h, d_e), dtype=np.float32 - input-to-hidden weights
    W_h: np.ndarray of shape (d_h, d_h), dtype=np.float32 - hidden-to-hidden weights
    b: np.ndarray of shape (d_h,), dtype=np.float32 - bias vector
    W_o: np.ndarray of shape (V, d_h), dtype=np.float32 - hidden-to-output weights
    b_o: np.ndarray of shape (V,), dtype=np.float32 - output bias
    returns: tuple (h_all, logits_all) where:
        - h_all: np.ndarray of shape (T, d_h)
        - logits_all: np.ndarray of shape (T, V)
    """
    T=len(x_chars)
    V=W_o.shape[0]
    
    d_h=W_h.shape[0]
    h_prev=np.zeros((d_h))
    h_all=np.zeros((T,d_h),dtype=np.float32)
    logits_all=np.zeros((T,V),dtype=np.float32)
    for t in range(1,T):
        if(isinstance(x_chars[t],np.int64)):
            id_t=x_chars[t]
        else:
            id_t=char_to_idx[x_chars[t]]
        e_t=embedding_weight[id_t]
        h_t=np.tanh(np.matmul(W_x,e_t)+np.matmul(W_h,h_prev)+b)
        o_t=np.matmul(h_t,W_o.T)+b_o
        h_all[t]=h_t
        logits_all[t]=o_t
        h_prev=h_t
    return (h_all,logits_all)
    raise NotImplementedError