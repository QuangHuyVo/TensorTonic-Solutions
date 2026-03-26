import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.

    Returns:
        np.ndarray of shape (n_test, k), dtype=int
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # Make sure both are 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # If no training points, return all -1
    if n_train == 0:
        return np.full((n_test, k), -1, dtype=int)

    # Squared Euclidean distances, fully vectorized
    distances = np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2)

    # Sort indices by distance
    sorted_idx = np.argsort(distances, axis=1)

    # If k is within range
    if k <= n_train:
        return sorted_idx[:, :k].astype(int)

    # If k is larger than n_train, pad with -1
    result = np.full((n_test, k), -1, dtype=int)
    result[:, :n_train] = sorted_idx
    return result