import numpy as np

def tanh(x):
    x = np.array(x)   # convert list → array
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))