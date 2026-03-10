import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):

    N = len(seqs)

    if N == 0:
        return np.zeros((0, 0), dtype=float)

    if max_len is None:
        L = max(len(seq) for seq in seqs)
    else:
        L = max_len

    out = np.full((N, L), pad_value, dtype=float)

    for i, seq in enumerate(seqs):
        seq = np.asarray(seq, dtype=float)
        length = min(len(seq), L)
        out[i, :length] = seq[:length]

    return out