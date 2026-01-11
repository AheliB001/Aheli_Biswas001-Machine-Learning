import numpy as np

theta = np.array([
    [2],
    [3],
    [3]
])

X = np.array([
    [1,0,2],
    [0,1,1],
    [2,1,0],
    [1,1,1],
    [0,2,1]
])

X_theta = X @ theta
print(X_theta)