import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)

    if y.size == 0:
        return 0.0

    # counts of each class
    _, counts = np.unique(y, return_counts=True)

    # probabilities
    p = counts / counts.sum()

    # avoid log(0)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)

    # entropy
    entropy = -np.sum(p * np.log2(p))

    return entropy