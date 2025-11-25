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


def resample_phonemes(cnp.int32_t[:] phonemes_arr, int target_length):
    cdef int source_length = phonemes_arr.shape[0]
    
    # Result array
    cdef cnp.int32_t[:] resampled = np.zeros(target_length, dtype=np.int32)
    
    # Circular-ish buffer to store pending phonemes.
    # We use a fixed size array large enough to hold phonemes (worst case: source_length)
    cdef cnp.int32_t[:] buffer = np.zeros(source_length, dtype=np.int32)
    cdef int buf_head = 0
    cdef int buf_tail = 0
    
    cdef int i, j
    cdef int start, end
    cdef double factor = <double>source_length / target_length
    
    cdef cnp.int32_t p
    cdef cnp.int32_t last_segment_p
    
    for i in range(target_length):
        # Calculate window edges (equivalent to np.linspace logic)
        start = <int>round(i * factor)
        end = <int>round((i + 1) * factor)
        
        if start == end:
            end += 1
            
        # Safety clamp
        if end > source_length:
            end = source_length
        
        # 1. Extract non-zero phonemes from this segment
        last_segment_p = -1
        
        for j in range(start, end):
            p = phonemes_arr[j]
            if p != 0:
                # Compress contiguous duplicates within the segment
                if p != last_segment_p:
                    last_segment_p = p
                    
                    # 2. Add to buffer
                    # Check against the last item currently in buffer to avoid merging across boundaries
                    # (Matches: if len(unused_buffer) == 0 or unused_buffer[-1] != p)
                    if buf_tail > buf_head:
                        if buffer[buf_tail - 1] != p:
                            buffer[buf_tail] = p
                            buf_tail += 1
                    else:
                        # Buffer is empty, just add
                        buffer[buf_tail] = p
                        buf_tail += 1
        
        # 3. Assign to current frame from buffer
        if buf_head < buf_tail:
            resampled[i] = buffer[buf_head]
            buf_head += 1
        else:
            resampled[i] = 0
            
    return np.asarray(resampled)

