from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, :2]   #first 2 features
y = iris.target

#Keep only classes 1 and 2
mask = y != 0
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# RBF Kernel
rbf_model = SVC(kernel='rbf', gamma=0.5)
rbf_model.fit(X_train, y_train)
rbf_pred = rbf_model.predict(X_test)

# Polynomial Kernel
poly_model = SVC(kernel='poly', degree=2)
poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)

# Accuracy comparison
print("RBF Accuracy    :", accuracy_score(y_test, rbf_pred))
print("Polynomial Accuracy :", accuracy_score(y_test, poly_pred))