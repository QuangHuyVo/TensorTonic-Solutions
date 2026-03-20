import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.asarray(x, dtype = float)
    if x.ndim ==1:
        xi = x - np.max(x)
        exp_x = np.exp(xi)
        return exp_x / np.sum(exp_x)
    elif x.ndim ==2:
        xi = x - np.max(x, axis = 1, keepdims =True)
        exp_x = np.exp(xi)
        return exp_x / np.sum(exp_x, axis = 1, keepdims = True)
    pass