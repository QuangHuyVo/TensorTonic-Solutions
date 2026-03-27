import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N, D) or (N, C, H, W).

    Args:
        x: Input array of shape (N, D) or (N, C, H, W)
        gamma: Scale parameter, shape (D,) or (C,)
        beta: Shift parameter, shape (D,) or (C,)
        eps: Small constant for numerical stability

    Returns:
        out: Batch-normalized output with same shape as x
    """
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if x.ndim == 2:
        # x: (N, D), normalize over batch axis
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)

        out = gamma.reshape(1, -1) * x_hat + beta.reshape(1, -1)

    elif x.ndim == 4:
        # x: (N, C, H, W), normalize each channel over (N, H, W)
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)

        out = gamma.reshape(1, -1, 1, 1) * x_hat + beta.reshape(1, -1, 1, 1)

    else:
        raise ValueError("x must have shape (N, D) or (N, C, H, W)")

    return out