import numpy as np
from sklearn.svm import SVC

X = np.array([
    [6, 5], [6, 9], [8, 6], [8, 8], [8, 10],
    [9, 2], [9, 5], [10, 10], [10, 13],
    [11, 5], [11, 8], [12, 6], [12, 11],
    [13, 4], [14, 8]
])

y = np.array([
    0, 0, 1, 1, 1,
    0, 1, 1, 0,
    1, 1, 1, 0,
    0, 0
])

#RBF Kernel
rbf_model = SVC(kernel='rbf', gamma=0.1, C=1)
rbf_model.fit(X, y)

#Polynomial Kernel
poly_model = SVC(kernel='poly', degree=2, C=1)
poly_model.fit(X, y)

#Comparing accuracy
print("RBF Accuracy :", rbf_model.score(X, y))
print("Polynomial Accuracy :", poly_model.score(X, y))