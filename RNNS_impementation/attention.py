import numpy as np
def softmax(z):
    return np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z)),keepdims=True)
def bahdanau_attention(decoder_hidden, encoder_states, W1, W2, v):
    """
    decoder_hidden: np.ndarray of shape (d_h,), dtype=np.float32 - decoder hidden state
    encoder_states: np.ndarray of shape (T, d_h), dtype=np.float32 - encoder hidden states
    W1: np.ndarray of shape (d_a, d_h), dtype=np.float32 - weight matrix for decoder
    W2: np.ndarray of shape (d_a, d_h), dtype=np.float32 - weight matrix for encoder
    v: np.ndarray of shape (d_a,), dtype=np.float32 - attention weight vector
    returns: tuple (context_vector, attention_weights) where:
        - context_vector: np.ndarray of shape (d_h,)
        - attention_weights: np.ndarray of shape (T,)
    """
    T,d_h=encoder_states.shape
    
    e=np.zeros(T,dtype=np.float32)
    W_dec=W1@decoder_hidden
    for t in range(T):
      
        e_t=v.T@np.tanh(W_dec+W2@encoder_states[t])
        e[t]=e_t
        
    attention_weights=softmax(e)

    context_vector=np.zeros(d_h,dtype=np.float32)
    for t in range(T):
        context_vector+=attention_weights[t]*encoder_states[t]
    
    

    return (context_vector,attention_weights)
    raise NotImplementedError