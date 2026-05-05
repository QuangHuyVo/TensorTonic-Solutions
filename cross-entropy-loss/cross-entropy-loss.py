import numpy as np

def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # small value to avoid log(0)
    eps = 1e-15
    
    # pick predicted probabilities of correct classes
    probs = y_pred[np.arange(len(y_true)), y_true]
    
    # compute loss
    loss = -np.log(probs + eps)
    
    return np.mean(loss)