import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)

    if not 0 <= p < 1:
        raise ValueError("p must satisfy 0 <= p < 1")

    if rng is None:
        rand_vals = np.random.random(size=x.shape)
    else:
        rand_vals = rng.random(size=x.shape)

    keep_prob = 1.0 - p
    scale = 1.0 / keep_prob

    keep_mask = rand_vals < keep_prob
    dropout_pattern = keep_mask.astype(float) * scale
    output = x * dropout_pattern

    return output, dropout_pattern