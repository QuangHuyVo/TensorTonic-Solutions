import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports:
        x: (D,) or (N,D)
        h_prev: (H,) or (N,H)
    """

    # Convert inputs to numpy arrays first
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)

    D = x.shape[-1]
    H = h_prev.shape[-1]

    x, x_was_1d = _as2d(x, D)
    h_prev, h_was_1d = _as2d(h_prev, H)

    # Update gate
    z = _sigmoid(
        x @ params["Wz"] +
        h_prev @ params["Uz"] +
        params["bz"]
    )

    # Reset gate
    r = _sigmoid(
        x @ params["Wr"] +
        h_prev @ params["Ur"] +
        params["br"]
    )

    # Candidate hidden state
    h_tilde = np.tanh(
        x @ params["Wh"] +
        (r * h_prev) @ params["Uh"] +
        params["bh"]
    )

    # Final hidden state
    h = (1 - z) * h_prev + z * h_tilde

    # Convert back to 1D if original input was 1D
    if x_was_1d and h_was_1d:
        return h[0]

    return h