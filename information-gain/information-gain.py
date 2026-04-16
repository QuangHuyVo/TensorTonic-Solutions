import numpy as np

def _entropy(y):
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    y = np.asarray(y)
    split_mask = np.asarray(split_mask)
    
    n = y.size
    if n == 0:
        return 0.0
    
    y_left = y[split_mask]
    y_right = y[~split_mask]
    
    H_parent = _entropy(y)
    H_left = _entropy(y_left)
    H_right = _entropy(y_right)
    
    n_left = y_left.size
    n_right = y_right.size
    
    weighted = (n_left / n) * H_left + (n_right / n) * H_right
    
    return H_parent - weighted
