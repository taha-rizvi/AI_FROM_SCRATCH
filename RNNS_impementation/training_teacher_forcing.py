def softmax(z):
    return np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z)))
import numpy as np
def encoder(x_seq,W_x,W_h,b):
    (T,d_x)=x_seq.shape
    d_h=W_h.shape[0]
    h_prev=np.zeros(d_h)
    h_all=np.zeros((T,d_h))
    for t in range(T):
        h_t=np.tanh(np.matmul(x_seq[t],W_x.T)+np.matmul(h_prev,W_h)+b)
        h_all[t]=h_t
        h_prev=h_t
    h_final=h_all[-1]
    return h_final.astype(np.float32)

def decoder(y_seq,W_x,W_h,b,W_o,b_o,context):

    d_h=W_h.shape[0]
    V=W_o.shape[0]
    MAX_T = 10  # or get from test context
    
    if y_seq is None:
        # Inference mode: generate T steps from start token
        T = MAX_T
        # Start with zero embedding or BOS token embedding
        y_seq = np.zeros((T, W_x.shape[1]), dtype=np.float32)
        # Force teacher_forcing=False for pure generation
        teacher_forcing = False
    else:
        (T, d_y) = y_seq.shape
        y_seq = y_seq
    
    if context.shape[0]==d_h:
        s_prev=context.copy()
    else:
        s_prev=np.zeros(d_h,dtype=np.float32)
    h_all=np.zeros((T,d_h),dtype=np.float32)
    o_all=np.zeros((T,V),dtype=np.float32)
    for t in range(T):
        s_t=np.tanh(np.matmul(y_seq[t],W_x.T)+np.matmul(s_prev,W_h)+b)
        o_t=np.matmul(s_t,W_o.T)+b_o
        p_t=softmax(o_t)
        o_all[t]=o_t
        h_all[t]=s_t
        s_prev=s_t
    return (h_all.astype(np.float32),o_all.astype(np.float32))




def teacher_forcing_training_step(x_seq, y_seq, encoder_params, decoder_params, loss_fn, optimizer_fn):
    """
    x_seq: np.ndarray of shape (T, d_x), dtype=np.float32 - input sequence
    y_seq: np.ndarray of shape (T', d_y), dtype=np.float32 - target sequence
    encoder_params: dict with keys {'W_x', 'W_h', 'b'} - encoder parameters
    decoder_params: dict with keys {'W_x', 'W_h', 'b', 'W_o', 'b_o'} - decoder parameters
    loss_fn: callable - function(logits, targets) -> scalar loss
    optimizer_fn: callable - function(params, grads) -> updated_params dict
    returns: dict with keys {'loss', 'predictions', 'updated_encoder_params', 'updated_decoder_params'}
    """
    context=encoder(x_seq,W_x,W_h,b)
    (h_all,logits)=decoder(y_seq,W_x,W_h,b,W_o,b_o)
    loss=loss_fn(logits,y_seq)
    grads=

    loss=loss_fn()

    raise NotImplementedError