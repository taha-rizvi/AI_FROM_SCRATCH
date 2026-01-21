import numpy as np
def log_softmax(z):
    z = z - np.max(z)
    return z - np.log(np.sum(np.exp(z)))

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
    beam=[([sos_token], context, 0.0)]
    for t in range(max_length):
        all_candidates=[]
        for sequence,hidden_state,score in beam:
            input_token=sequence[-1]
            if input_token==eos_token or len(sequence) >= max_length:
                all_candidates.append((sequence, hidden_state, score))
                continue
            next_hidden, logits = decoder_fn(hidden_state, input_token)
            log_probs=log_softmax(logits)
            top_k_idx=np.argpartition(-log_probs,beam_width-1)[:beam_width] #argpartition returns indices of the top k elements it actually creates a an array with first k+1 elements sorted in increasing order rest are arbitrary
           
            
            for idx in top_k_idx:
               
                new_seq=sequence+[idx]
                new_score=score+log_probs[idx]
                
                all_candidates.append((new_seq,next_hidden,new_score))
            
            
        beam=sorted(all_candidates,key=lambda x:x[2],reverse=True)[:beam_width] 
        # np.all is used to check if all sequences have ended
        if all(s[-1]==eos_token for s,_,_ in beam): break
    final_results=[]
    for sequence,hidden_state,score in beam:
        penalized_score=score/(len(sequence)**length_penalty)
        if sequence[-1]==eos_token:
            final_results.append((sequence, penalized_score.astype(np.float64)))
        elif len(sequence) == max_length:  # Unfinished but max_length reached
            final_results.append((sequence,penalized_score.astype(np.float64)))
    final_results=sorted(final_results,key=lambda x:x[1],reverse=True)
    return final_results


    raise NotImplementedError

 