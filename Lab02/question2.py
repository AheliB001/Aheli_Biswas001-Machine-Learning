import numpy as np

x = np.array([
    [2],
    [1],
    [2]
])
y = np.array([
    [1],
    [2],
    [2]
])

dot = 0
for i in range(len(x)):
    dot = dot + (x[i] * y[i]).item()
print(dot)
