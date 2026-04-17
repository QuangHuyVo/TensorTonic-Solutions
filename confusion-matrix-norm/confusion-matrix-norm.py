import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # infer classes if needed
    if num_classes is None:
        if y_true.size == 0 and y_pred.size == 0:
            num_classes = 0
        else:
            num_classes = int(max(
                y_true.max() if y_true.size else 0,
                y_pred.max() if y_pred.size else 0
            )) + 1
    
    # create matrix
    C = np.zeros((num_classes, num_classes), dtype=float)
    
    # only fill if not empty
    if y_true.size > 0:
        np.add.at(C, (y_true, y_pred), 1)
    
    if normalize == 'none':
        return C
    
    if normalize == 'true':
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return C / row_sums
    
    if normalize == 'pred':
        col_sums = C.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        return C / col_sums
    
    if normalize == 'all':
        total = C.sum()
        return C / total if total > 0 else C
    
    raise ValueError("normalize must be one of: 'none', 'true', 'pred', 'all'")