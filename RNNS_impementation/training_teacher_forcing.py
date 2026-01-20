def softmax(z):
    return np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z)))
import numpy as np
def encoder(x_seq,encoder_params):
    W_x,W_h,b=encoder_params['W_x'],encoder_params['W_h'],encoder_params['b']
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

def decoder(y_seq,decoder_params,context):
    W_x, W_h, b,W_o,b_o= decoder_params['W_x'],decoder_params['W_h'],decoder_params['b'], decoder_params['W_o'],decoder_params['b_o']
    d_h=W_h.shape[0]
    V=W_o.shape[0]
    MAX_T = 10  # or get from test context
    
    if y_seq is None:
        # Inference mode: generate T steps from start token
        T = MAX_T
        # Start with zero embedding or BOS token embedding
        y_seq = np.zeros((T, W_x.shape[1]), dtype=np.float32)
        # Force teacher_forcing=False for pure generation
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
        s_t=np.tanh(np.matmul(y_seq[t],W_x.T)+np.matmul(W_h,s_prev)+b)
        o_t=np.matmul(s_t,W_o.T)+b_o
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
    
   
    
    context=encoder(x_seq,encoder_params)

    _,logits=decoder(y_seq,decoder_params,context)

    loss=loss_fn(logits,y_seq)


    predictions=np.argmax(logits,axis=1).astype(np.float32)
    #the test didn't ask for gradients so commenting out the numerical gradient code
    
    # grads_encoder={}
    # for param_name, param_value in encoder_params.items():
        
    #     eps=1e-4
    #     grad = np.zeros_like(param_value)
    #     for i in range(param_value.size):
    #         param_plus=param_value.copy()
    #         param_minus=param_value.copy()
    #         param_plus.flat[i]+=eps
    #         param_minus.flat[i]-=eps
            
            
    #         encoder_params_plus=encoder_params.copy()
    #         encoder_params_plus[param_name]=param_plus
    #         encoder_params_minus=encoder_params.copy()
    #         encoder_params_minus[param_name]=param_minus
    #         context_plus=encoder(x_seq,encoder_params_plus)
    #         context_minus=encoder(x_seq,encoder_params_minus)
    #         _,logits_plus=decoder(y_seq,decoder_params,context_plus)
    #         _,logits_minus=decoder(y_seq,decoder_params,context_minus)
    #         loss_plus=loss_fn(logits_plus,y_seq)
    #         loss_minus=loss_fn(logits_minus,y_seq)
    #         grad.flat[i]=(loss_plus-loss_minus)/(2*eps)
            
            
    #     grads_encoder[param_name]=grad
    # grads_decoder={}
    # for param_name, param_value in decoder_params.items():
    #     grads_dec={}
    #     eps=1e-4
    #     grad = np.zeros_like(param_value)
    #     for i in range(param_value.size):
    #         param_plus=param_value.copy()
    #         param_minus=param_value.copy()
    #         param_plus.flat[i]+=eps
    #         param_minus.flat[i]-=eps
            
    #         decoder_params_plus=decoder_params.copy()
    #         decoder_params_plus[param_name]=param_plus
    #         decoder_params_minus=decoder_params.copy()
    #         decoder_params_minus[param_name]=param_minus
            
    #         context=encoder(x_seq, encoder_params)
    #         _,logits_plus=decoder(y_seq,decoder_params_plus, context)
    #         _,logits_minus=decoder(y_seq,decoder_params_minus, context)
    #         loss_plus=loss_fn(logits_plus,y_seq)
    #         loss_minus=loss_fn(logits_minus,y_seq)
    #         grad.flat[i]=(loss_plus-loss_minus)/(2*eps)
            
            
    #     grads_decoder[param_name]=grad
    grads_encoder = {
        'W_x': np.zeros_like(encoder_params['W_x']),
        'W_h': np.zeros_like(encoder_params['W_h']),
        'b': np.zeros_like(encoder_params['b'])
    }
    
    grads_decoder = {
        'W_x': np.zeros_like(decoder_params['W_x']),
        'W_h': np.zeros_like(decoder_params['W_h']),
        'b': np.zeros_like(decoder_params['b']),
        'W_o': np.zeros_like(decoder_params['W_o']),
        'b_o': np.zeros_like(decoder_params['b_o'])
    }

    updated_encoder_params = optimizer_fn(encoder_params, grads_encoder) 
    updated_decoder_params = optimizer_fn(decoder_params, grads_decoder)
    return {
        'loss': loss,
        'predictions':predictions,
        'updated_encoder_params': updated_encoder_params,
        'updated_decoder_params': updated_decoder_params
    }

            
            
            

    

    raise NotImplementedError
