import numpy as np
def softmax(z):
    return np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z),keepdims=True))
def luong_attention(v,decoder_hidden, encoder_states, W=None, score_type='general'):
    """
    decoder_hidden: np.ndarray of shape (d_h,), dtype=np.float32 - decoder hidden state
    encoder_states: np.ndarray of shape (T, d_h), dtype=np.float32 - encoder hidden states
    W: None or np.ndarray of shape (d_h, d_h), dtype=np.float32 - weight matrix for general scoring
    score_type: str - one of 'dot', 'general', 'concat'
    returns: tuple (context_vector, attention_weights) where:
        - context_vector: np.ndarray of shape (d_h,)
        - attention_weights: np.ndarray of shape (T,)
    """
    T,d_h=encoder_states.shape

    e=np.zeros(T,dtype=np.float32)
    if score_type=='dot':
        for t in range(T):
            e_t=np.dot(decoder_hidden.T,encoder_states[t])
            e[t]=e_t
    if score_type=='general':
        for t in range(T):
            e_t=np.dot(decoder_hidden.T,np.dot(W,encoder_states[t]))
            e[t]=e_t
    if score_type=='concat':
        for t in range(T):
            e_t=np.dot(v.T,np.dot(W[:,:d_h],decoder_hidden)+np.dot(W[:,d_h:],encoder_states[t]))
            e[t]=e_t
    attention_weight=softmax(e)
    context_vector=np.zeros(d_h,dtype=np.float32)
    for t in range(T):
        context_vector+=attention_weight[t]*encoder_states[t]
    return (context_vector,attention_weight)
    raise NotImplementedError
