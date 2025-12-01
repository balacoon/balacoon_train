import numpy as np
cimport numpy as cnp
from libc.math cimport round

def resample_pitch(cnp.float32_t[:] pitch_arr, int target_length):
    """
    Downsamples pitch array to target_length.
    Calculates the mean of non-zero (voiced) values for each target frame 
    to avoid smearing unvoiced (0) regions into voiced ones.
    """
    cdef int source_length = pitch_arr.shape[0]
    cdef cnp.float32_t[:] resampled = np.zeros(target_length, dtype=np.float32)
    
    cdef int i, j, k
    cdef int start, end
    cdef int segment_len
    cdef double factor = <double>source_length / target_length
    
    # Buffer for median calculation. 
    cdef cnp.float32_t[:] segment_buf = np.zeros(source_length, dtype=np.float32)
    
    cdef int non_zero_count
    cdef float val, tmp
    
    for i in range(target_length):
        start = <int>round(i * factor)
        end = <int>round((i + 1) * factor)
        
        if start == end:
            end += 1
            
        if end > source_length:
            end = source_length
            
        segment_len = end - start
        if segment_len <= 0:
            resampled[i] = 0.0
            continue
        
        non_zero_count = 0
        # Collect segment values
        for j in range(start, end):
            val = pitch_arr[j]
            segment_buf[j - start] = val
            if val != 0:
                non_zero_count += 1
        
        # Logic: if majority of segment is unvoiced (0), set to 0
        if <double>non_zero_count / segment_len < 0.5:
            resampled[i] = 0.0
        else:
            # Calculate median of the segment
            # Simple Bubble Sort
            for j in range(segment_len):
                for k in range(j + 1, segment_len):
                    if segment_buf[j] > segment_buf[k]:
                        tmp = segment_buf[j]
                        segment_buf[j] = segment_buf[k]
                        segment_buf[k] = tmp
            
            # Median
            if segment_len % 2 == 1:
                resampled[i] = segment_buf[segment_len // 2]
            else:
                resampled[i] = 0.5 * (segment_buf[segment_len // 2 - 1] + segment_buf[segment_len // 2])
                
    return np.asarray(resampled)


def resample_phonemes(cnp.float32_t[:, :] phoneme_probs, cnp.int32_t[:, :] phoneme_indices, int target_length, int vocab_size):
    """
    Resamples phoneme probabilities to target_length.
    Takes sparse representation (probs + indices) and accumulates them into dense matrix.
    """
    cdef int source_length = phoneme_probs.shape[0]
    cdef int num_hyps = phoneme_probs.shape[1]
    
    # Result array (T x vocab_size)
    cdef cnp.float32_t[:, :] resampled = np.zeros((target_length, vocab_size), dtype=np.float32)
    
    cdef int i, j, k
    cdef int start, end
    cdef double factor = <double>source_length / target_length
    cdef int idx
    cdef float val
    
    for i in range(target_length):
        start = <int>round(i * factor)
        end = <int>round((i + 1) * factor)
        
        if start == end:
            end += 1
            
        if end > source_length:
            end = source_length
            
        # Sum probabilities from source frames
        for j in range(start, end):
            for k in range(num_hyps):
                idx = phoneme_indices[j, k]
                val = phoneme_probs[j, k]
                if idx >= 0 and idx < vocab_size:
                    resampled[i, idx] += val
            
    return np.asarray(resampled)

