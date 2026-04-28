import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    
    predictions: array of predicted probabilities (0 to 1)
    targets: array of binary labels (0 or 1)
    alpha: balancing factor
    gamma: focusing parameter
    """
    preds = np.asarray(predictions, dtype=float)
    t = np.asarray(targets, dtype=float)
    
    # Avoid log(0)
    eps = 1e-12
    preds = np.clip(preds, eps, 1 - eps)
    
    # p_t: probability of the true class
    p_t = np.where(t == 1, preds, 1 - preds)
    
    # focal loss per sample
    loss = -alpha * ((1 - p_t) ** gamma) * np.log(p_t)
    
    return np.mean(loss)