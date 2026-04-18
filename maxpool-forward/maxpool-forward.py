import numpy as np

def maxpool_forward(X, pool_size, stride):
    X = np.array(X)
    
    H, W = X.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    
    out = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            
            window = X[h_start:h_end, w_start:w_end]
            out[i, j] = np.max(window)
    
    return out.tolist()