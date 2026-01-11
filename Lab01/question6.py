#Implement y = 2x1 + 3x2 + 3x3 + 4, where x1, x2 and x3 are three independent variables. Compute the gradient of y at a few points and print the values.

import numpy as np
import matplotlib.pyplot as p

points = [
    (9,5,7),
    (1,2,3),
    (4,5,6)
]

dy_dx1 = 2
dy_dx2 = 3
dy_dx3 = 3

Gradient = np.array([
    [2],
    [3],
    [3]
])
for x1, x2, x3 in points:
    y = 2*x1 + 3*x2 + 3*x3 + 4

    print("Point:", x1, x2, x3)
    print("y =", y)
    print("Gradient =", Gradient)
