import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.array(g, dtype=float)
    
    # Special case: no clipping
    if max_norm <= 0:
        return g
    
    norm = np.linalg.norm(g)
    
    if norm > max_norm:
        g = g * (max_norm / norm)
    
    return g