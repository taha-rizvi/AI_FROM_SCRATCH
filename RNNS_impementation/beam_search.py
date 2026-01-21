import numpy as np 
def log_softmax(z):
    return np.log(np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z))))

def beam_search_decode(context, decoder_fn, vocab_size, beam_width=3, max_length=50, sos_token=0, eos_token=1, length_penalty=0.0):
    """
    context: np.ndarray of shape (d_c,), dtype=np.float32 - encoder context vector
    decoder_fn: callable - function(hidden_state, input_token) -> (next_hidden, logits)
        hidden_state: np.ndarray of shape (d_h,), dtype=np.float32
        input_token: int - token index
        returns: tuple (next_hidden, logits) where:
            next_hidden: np.ndarray of shape (d_h,), dtype=np.float32
            logits: np.ndarray of shape (V,), dtype=np.float32
    vocab_size: int - vocabulary size V
    beam_width: int - number of candidates to maintain
    max_length: int - maximum sequence length
    sos_token: int - start-of-sequence token index
    eos_token: int - end-of-sequence token index
    length_penalty: float - length normalization exponent alpha
    returns: list of tuples [(sequence, score), ...] sorted by score (descending)
        sequence: list of ints - token indices
        score: float - log probability score
    """
    beam=[(sos_token, context, 0.0)]
    for t in range(max_length):
        all_candidates=[]
        for parent in beam:
            input_token, hidden_state, score =parent[0], parent[1], parent[2]
            next_hidden, logits = decoder_fn(hidden_state, input_token)
            log_probs=log_softmax(logits)
            top_k_idx=np.argpartition(-log_probs,beam_width-1)[:beam_width]
            # top_k_scores=log_probs[top_k_idx]
            # order=np.argsort(-top_k_vals)
            # top_k_scores=top_k_vals[order]
            # top_k_idx=top_k_idx[order]
            
            for idx in top_k_idx:

                input_token+=idx
                score+=log_probs[idx]
                hidden_state=next_hidden
                all_candidates.append((input_token,hidden_state,score))
            
            all_candidates=sorted(all_candidates,key=lambda x:x[2], reverse=True)
        beam=all_candidates[:beam_width]


    raise NotImplementedError

 