#Implement ATA  -  A = [1 2 3
# 4 5 6]

import numpy as np

A = np.array([
    [1,2,3],
    [4,5,6]
])

ATA = A.T @ A   #@ means matrix multiplication

print("A:")
print(A)

print("A^T A:")
print(ATA)