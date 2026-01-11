#Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100].

import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(-10,10, 100)
y = x1**2
plt.plot(x1, y)
plt.xlabel("x1")
plt.ylabel("y")
plt.grid(True)
plt.show()

#dx/dy(y) = 2 * x1

points = np.array([-5, -3, 0, 3, 5])
derivates_at_points = 2 * points
print(f"Derivates at", points, "are", derivates_at_points, "respectively.")

#What is the value of x1 at which the function value (y) is zero. What do you infer from this?
x1_0 = 0
print("Function value y = 0 at x1 =", x1_0)

