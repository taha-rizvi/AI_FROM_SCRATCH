import numpy as np
def seq2seq_encoder(x_seq, W_x, W_h, b, cell_type='rnn', bidirectional=False):
    """
    x_seq: np.ndarray of shape (T, d_x), dtype=np.float32 - input sequence
    W_x: np.ndarray of shape (d_h, d_x), dtype=np.float32 - input-to-hidden weights
    W_h: np.ndarray of shape (d_h, d_h), dtype=np.float32 - hidden-to-hidden weights
    b: np.ndarray of shape (d_h,), dtype=np.float32 - bias vector
    cell_type: str - one of 'rnn', 'lstm', 'gru'
    bidirectional: bool - if True, process in both directions
    returns: tuple (h_all, h_final) where:
        - h_all: np.ndarray of shape (T, d_h) or (T, 2*d_h) if bidirectional
        - h_final: np.ndarray of shape (d_h,) or (2*d_h,) if bidirectional
    """
    (T,d_h)=x_seq.shape
    if(bidirectional):
        if cell_type=='rnn': 
            h_fwd=np.zeros((T,d_h))
            h_prev=np.zeros(d_h)
            for t in range(T):
                h_t=np.tanh(np.matmul(x_seq[t],W_x.T)+np.matmul(h_prev,W_h.T))
                h_fwd[t]=h_t
                h_prev=h_t
            h_fwd_final=h_fwd[-1]
            h_bwd=np.zeros((T,d_h))
            h_prev=np.zeros(d_h)
            for t in reversed(range(T)):
                h_t=np.tanh(np.matmul(x_seq[t],W_x.T)+np.matmul(h_prev,W_h.T))
                h_bwd[t]=h_t
                h_prev=h_t
            h_bwd_final=h_bwd[0]
            h_ans_all=np.concat(h_fwd,h_bwd,axis=1)
            h_ans_final=np.concat(h_fwd_final,h_bwd_final,axis=1)
            return (h_ans_all,h_ans_final)  
        if cell_type=='lstm':
            h_fwd=np.zeros((T,d_h))
            h_prev=np.zeros(d_h)
            c_prev=np.zeros(d_h)
            for t in range(T):
                z_t=np.matmul(x_seq[t],W_x.T)+np.matmul(h_prev,W_h.T)+b
                z_f,z_i,z_c,z_o=np.split(z_t,4)
                f_t=1/(1+np.exp(-z_f))
                i_t=1/(1+np.exp(-z_i))
                c_hat_t=np.tanh(z_c)
                o_t=1/(1+np.exp(-z_o))
                c_t=f_t*c_prev+i_t*c_hat_t
                h_t=o_t*np.tanh(c_t)
                h_fwd[t]=h_t
                c_prev=c_t
                h_prev=h_t
            h_fwd_final=h_fwd[-1]
            h_bwd=np.zeros((T,d_h))
            h_prev=np.zeros(d_h)
            c_prev=np.zeros(d_h)
            for t in reversed(range(T)):
                z_t=np.matmul(x_seq[t],W_x.T)+np.matmul(h_prev,W_h.T)+b
                z_f,z_i,z_c,z_o=np.split(z_t,4)
                f_t=1/(1+np.exp(-z_f))
                i_t=1/(1+np.exp(-z_i))
                c_hat_t=np.tanh(z_c)
                o_t=1/(1+np.exp(-z_o))
                c_t=f_t*c_prev+i_t*c_hat_t
                h_t=o_t*np.tanh(c_t)
                h_bwd[t]=h_t
                c_prev=c_t
                h_prev=h_t
            h_bwd_final=h_bwd[0]
            h_ans_all=np.concat(h_fwd,h_bwd,axis=1)
            h_ans_final=np.concat(h_fwd_final,h_bwd_final,axis=1)
            return(h_ans_all,h_ans_final)
        if cell_type=='gru':
            h_fwd=np.zeros((T,d_h))
            h_prev=np.zeros(d_h)
            for t in range(T):
                z_t=np.matmul(x_seq[t],W_x.T)+np.matmul(h_prev,W_h.T)+b
                z_z,z_r,z_h=np.split(z_t,3)
                u_t=1/(1+np.exp(-z_z))
                r_t=1/(1+np.exp(-z_r))
                h_hat=np.tanh(z_h*r_t)
                h_t=(1-u_t)*h_hat+u_t*h_prev
                h_fwd[t]=h_t
            h_fwd_final=h_fwd[-1]
            h_bwd=np.zeros((T,d_h))
            h_prev=np.zeros(d_h)
            for t in reversed(range(T)):
                z_t=np.matmul(x_seq[t],W_x.T)+np.matmul(h_prev,W_h.T)+b
                z_z,z_r,z_h=np.split(z_t,3)
                u_t=1/(1+np.exp(-z_z))
                r_t=1/(1+np.exp(-z_r))
                h_hat=np.tanh(z_h*r_t)
                h_t=(1-u_t)*h_hat+u_t*h_prev
                h_bwd[t]=h_t
            h_bwd_final=h_bwd[0]
            h_ans_all=np.concat(h_fwd,h_bwd,axis=1)
            h_ans_final=np.concat(h_fwd_final,h_bwd_final,axis=1)
            return(h_ans_all,h_ans_final)
    else:
        if cell_type=='rnn':
        
            h_all=np.array(np.zeros((T,d_h)))
            h_prev=np.zeros(d_h)
        
            for t in range(T):
                h_t=np.tanh(np.matmul(x_seq[t],W_x.T)+np.matmul(h_prev,W_h.T))
                h_all[t]=h_t
                h_prev=h_t
            
            h_final=h_all[-1]
            return (h_all,h_final)
        if cell_type=='lstm':
            h_all=np.array(np.zeroes((T,d_h)))
            h_prev=np.zeros(d_h)
            c_prev=np.zeros(d_h)
            for t in range(T):
                z_t=np.matmul(x_seq[t],W_x.T)+np.matmul(h_prev,W_h.T)+b
                z_f,z_i,z_c,z_o=np.split(z_t,4)
                f_t=1/(1+np.exp(-z_f))
                i_t=1/(1+np.exp(-z_i))
                c_hat_t=np.tanh(z_c)
                o_t=1/(1+np.exp(-z_o))
                c_t=f_t*c_prev + i_t*c_hat_t
                h_t=o_t*np.tanh(c_t)
                h_all[t]=h_t
                c_prev=c_t
                h_prev=h_t
            h_final=h_all[-1]
            return (h_all,h_final)    
        if cell_type=='gru':
            h_all=np.array(np.zeroes((T,d_h)))
            h_prev=np.zeroes(d_h)
            for t in range(T):
                z_t=np.matmul(x_seq[t],W_x.T)+np.matmul(h_prev,W_h.T)+b
                z_z,z_r,z_h=np.split(z_t,3)
                u_t=1/(1+np.exp(-z_z))
                r_t=1/(1+np.exp(-z_r))
                h_hat=np.tanh(z_h*r_t)
                h_t=(1-u_t)*h_hat+u_t*h_prev
                h_all[t]=h_t
            h_final=h_all[-1]
            return (h_all,h_final)
            



    raise NotImplementedError