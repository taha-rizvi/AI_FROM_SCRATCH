import numpy as np 
def softmax(z):
    return np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z)),axis=1).reshape(-1,1)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def seq2seq_decoder(context, y_seq, W_x, W_h, b, W_o, b_o, cell_type='rnn', teacher_forcing=True):
    """
    context: np.ndarray of shape (d_c,), dtype=np.float32 - encoder context vector
    y_seq: np.ndarray of shape (T', d_y) or None, dtype=np.float32 - target sequence
    W_x: np.ndarray of shape (d_h, d_y), dtype=np.float32 - input-to-hidden weights
    W_h: np.ndarray of shape (d_h, d_h), dtype=np.float32 - hidden-to-hidden weights
    b: np.ndarray of shape (d_h,), dtype=np.float32 - bias vector
    W_o: np.ndarray of shape (V, d_h), dtype=np.float32 - hidden-to-output weights
    b_o: np.ndarray of shape (V,), dtype=np.float32 - output bias
    cell_type: str - one of 'rnn', 'lstm', 'gru'
    teacher_forcing: bool - if True, use ground truth tokens; if False, use predictions
    returns: tuple (h_all, logits_all) where:
        - h_all: np.ndarray of shape (T', d_h)
        - logits_all: np.ndarray of shape (T', V)
    """
    (T,d_y)=y_seq.shape
    d_h=W_h.shape[0]
    V=W_o.shape[0]
    if(teacher_forcing):
        if cell_type=='rnn':
            s_prev=context.copy()
            y_prev=y_seq[0]
            h_all=np.zeros((T,d_h))
            o_all=np.zeros((T,V))
            for t in range(1,T):
                s_t=np.tanh(np.matmul(y_prev,W_x.T)+np.matmul(W_h,s_prev)+b) 
                o_t=np.matmul(s_t,W_o.T)+b_o
                p_t=softmax(o_t)
                o_all[t]=o_t
                h_all[t]=s_t
                y_prev=y_seq[np.argmax(p_t)]
                s_prev=s_t
            return (h_all.astype(np.float32),o_all.astype(np.float32)) 
        if cell_type=='lstm':
            s_prev=context.copy()
            y_prev=y_seq[0]
            c_prev=0
            h_all=np.zeros((T,d_h))
            o_all=np.zeros((T,V))
            for t in range(1,T):
                z_t=np.tanh(np.matmul(y_prev,W_x.T)+np.matmul(W_h,s_prev)+b)
                z_f,z_i,z_c,z_o=np.split(z_t,4)
                f_t=sigmoid(z_f)
                i_t=sigmoid(z_i)
                c_hat_t=np.tanh(z_c)
                o=sigmoid(z_o)
                c_t=f_t*c_prev+i_t*c_hat_t
                s_t=o*np.tanh(c_t)
                o_t=np.matmul(s_t,W_o.T)+b_o
                p_t=softmax(o_t)
                h_all[t]=s_t
                o_all[t]=o_t
            return (h_all.astype(np.float32),o_all.astype(np.float32))
        if cell_type=='gru':
            s_prev=context.copy()
            y_prev=y_seq[0]
            c_prev=0
            h_all=np.zeros((T,d_h))
            o_all=np.zeros((T,V))
            for t in range(1,T):
                z_t=np.tanh(np.matmul(y_prev,W_x.T)+np.matmul(W_h,s_prev)+b)
                z_z,z_r,z_h=np.split(z_t,3)
                u_t=sigmoid(z_z)
                r_t=sigmoid(z_r)
                h_hat=np.tanh(z_h*r_t)
                s_t=(1-u_t)*h_hat+u_t*s_prev
                o_t=np.matmul(s_t,W_o.T)+b_o
                p_t=softmax(o_t)
                h_all[t]=s_t
                o_all[t]=o_t
            return (h_all.astype(np.float32),o_all.astype(np.float32))
    else:
        if cell_type=='rnn':
            s_prev=context.copy()
            y_prev=y_seq[0]
            h_all=np.zeros((T,d_h))
            o_all=np.zeros((T,V))
            for t in range(1,T):
                s_t=np.tanh(np.matmul(y_prev,W_x.T)+np.matmul(W_h,s_prev)+b) 
                o_t=np.matmul(s_t,W_o.T)+b_o
                p_t=softmax(o_t)
                o_all[t]=o_t
                h_all[t]=s_t
                h_prev=y_seq[np.argmax(p_t)]
                s_prev=s_t
            return (h_all.astype(np.float32),o_all.astype(np.float32)) 
        if cell_type=='lstm':
            s_prev=context.copy()
            y_prev=y_seq[0]
            c_prev=0
            h_all=np.zeros((T,d_h))
            o_all=np.zeros((T,V))
            for t in range(1,T):
                z_t=np.tanh(np.matmul(y_prev,W_x.T)+np.matmul(W_h,s_prev)+b)
                z_f,z_i,z_c,z_o=np.split(z_t,4)
                f_t=sigmoid(z_f)
                i_t=sigmoid(z_i)
                c_hat_t=np.tanh(z_c)
                o=sigmoid(z_o)
                c_t=f_t*c_prev+i_t*c_hat_t
                s_t=o*np.tanh(c_t)
                o_t=np.matmul(s_t,W_o.T)+b_o
                p_t=softmax(o_t)
                h_all[t]=s_t
                o_all[t]=o_t
                y_prev=y_seq[np.argmax(p_t)]
                s_prev=s_t
            return (h_all.astype(np.float32),o_all.astype(np.float32))
        if cell_type=='gru':
            s_prev=context.copy()
            y_prev=y_seq[0]
            c_prev=0
            h_all=np.zeros((T,d_h))
            o_all=np.zeros((T,V))
            for t in range(1,T):
                z_t=np.tanh(np.matmul(y_prev,W_x.T)+np.matmul(W_h,s_prev)+b)
                z_z,z_r,z_h=np.split(z_t,3)
                u_t=sigmoid(z_z)
                r_t=sigmoid(z_r)
                h_hat=np.tanh(z_h*r_t)
                s_t=(1-u_t)*h_hat+u_t*s_prev
                o_t=np.matmul(s_t,W_o.T)+b_o
                p_t=softmax(o_t)
                h_all[t]=s_t
                o_all[t]=o_t
                y_prev=y_seq[np.argmax(p_t)]
                s_prev=s_t

            return (h_all.astype(np.float32),o_all.astype(np.float32))
    raise NotImplementedError
        