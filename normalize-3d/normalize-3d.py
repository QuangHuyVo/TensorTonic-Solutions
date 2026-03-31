import numpy as np

def normalize_3d(v):
    v = np.asarray(v, dtype=float)

    # Compute norm
    norm = np.linalg.norm(v, axis=-1, keepdims=True)

    # Avoid division by zero
    # Where norm == 0, keep original (zeros)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm != 0)
    pass