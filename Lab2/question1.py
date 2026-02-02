import numpy as np

X = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
])

X_mean = np.mean(X, axis=0)
X_centered = X - X.mean(axis=0)
X_cT = X_centered.T

Y= X_cT @ X_centered

print(Y/4)

#checking with numpy
X_Centered = X - np.mean(X, axis=0)

#computing covariance matrix
cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)

print( f"Covariance Matrix:\n", cov_matrix)
