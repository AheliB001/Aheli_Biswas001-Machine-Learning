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
    dot = dot + (x[i].item() * y[i].item())
print(dot)


#dot product refers to the scalar multiplication between 2 given vectors, which produces a scalar quantity.
#example
a=np.array([
    [1],
    [2],
    [4]])

b=np.array([
    [5],
    [2],
    [1]])

dott = 0
for j in range(len(a)):
    dott = dott + (a[j].item() * b[j].item())
print(dott)


