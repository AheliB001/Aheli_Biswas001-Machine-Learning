import numpy as np

X = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
])

# Center the matrix
X_centered = X - np.mean(X, axis=0)

# Compute covariance matrix
cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)

print(cov_matrix)
