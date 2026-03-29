import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).

    Parameters
    ----------
    T : np.ndarray of shape (4, 4)
        Homogeneous transformation matrix
    points : np.ndarray of shape (3,) or (N, 3)
        Input 3D point(s)

    Returns
    -------
    np.ndarray
        Transformed point(s), shape (3,) if input was single point,
        else (N, 3)
    """
    points = np.asarray(points, dtype=float)

    # Detect single point
    single = (points.ndim == 1)
    if single:
        points = points.reshape(1, 3)

    # Convert to homogeneous coordinates
    N = points.shape[0]
    ones = np.ones((N, 1))
    points_h = np.hstack((points, ones))  # (N, 4)

    # Apply transformation
    transformed_h = (T @ points_h.T).T  # (N, 4)

    # Convert back to 3D
    transformed = transformed_h[:, :3]

    # Return original shape
    return transformed[0] if single else transformed


# Example usage
if __name__ == "__main__":
    T = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1]
    ])

    points = np.array([[0, 0, 0], [1, 1, 1]])

    result = apply_homogeneous_transform(T, points)
    print(result)