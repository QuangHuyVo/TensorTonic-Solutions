import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # True positives = correct predictions
    tp = np.sum(y_true == y_pred)

    # For single-label multi-class:
    # total predictions = total actual = len(y_true)
    total = y_true.size

    if total == 0:
        return 0.0

    precision = tp / total
    recall = tp / total

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)

    return float(f1)