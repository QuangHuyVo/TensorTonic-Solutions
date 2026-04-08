import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    G = np.asarray(G, dtype=float)

    # Step 1
    G = G + g**2

    # Step 2 (correct version for this problem)
    w = w - (lr / np.sqrt(G + eps)) * g

    return w, G