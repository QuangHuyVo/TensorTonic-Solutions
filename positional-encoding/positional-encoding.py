import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angles = pos / (base ** (2 * (i // 2) / d_model))

    pe = np.empty((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe