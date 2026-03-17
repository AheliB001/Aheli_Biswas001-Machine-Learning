from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale manually
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ridge
model = RidgeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("ridge accuracy score:", accuracy_score(y_test, y_pred))


# Lasso
lasso = LogisticRegression(penalty='l1', solver='liblinear')
lasso.fit(X_train, y_train)
print("lasso accuracy score:", accuracy_score(y_test, lasso.predict(X_test)))
