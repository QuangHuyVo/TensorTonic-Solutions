import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray (x)
    p = np.asarray (p, dtype = float)

    if np.allclose(np.sum(p), 1) is False:
        raise ValueError
    else: 
        Ex = np.average (x, weights = p)

    return Ex
    pass
