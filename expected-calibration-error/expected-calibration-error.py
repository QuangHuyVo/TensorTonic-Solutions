import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        
        # include right edge in last bin
        if i == n_bins - 1:
            mask = (y_pred >= left) & (y_pred <= right)
        else:
            mask = (y_pred >= left) & (y_pred < right)
        
        if np.sum(mask) == 0:
            continue
        
        bin_confidence = np.mean(y_pred[mask])
        bin_accuracy = np.mean(y_true[mask])
        
        ece += (np.sum(mask) / n) * abs(bin_accuracy - bin_confidence)
    
    return ece