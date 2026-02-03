import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)

    for i in range(iterations):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradient = (1 / m) * np.dot(X.T, errors)
        theta = theta - learning_rate * gradient

    return theta

df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

X = df.iloc[:, :5].values
y = df["disease_score_fluct"].values

mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

theta = gradient_descent(X_train, y_train, learning_rate=0.01, iterations=2000)
y_pred_gd = np.dot(X_test, theta)

lr = LinearRegression()
lr.fit(X_train[:, 1:], y_train)
y_pred_sk = lr.predict(X_test[:, 1:])

print("SIMULATED DATASET")
print("Gradient Descent MSE:", mean_squared_error(y_test, y_pred_gd))
print("Scikit-learn MSE   :", mean_squared_error(y_test, y_pred_sk))


X, y = fetch_california_housing(return_X_y=True)


mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

theta = gradient_descent(X_train, y_train, learning_rate=0.01, iterations=3000)
y_pred_gd = np.dot(X_test, theta)

lr = LinearRegression()
lr.fit(X_train[:, 1:], y_train)
y_pred_sk = lr.predict(X_test[:, 1:])

print("\nCALIFORNIA HOUSING DATASET")
print("Gradient Descent R2:", r2_score(y_test, y_pred_gd))
print("Scikit-learn R2   :", r2_score(y_test, y_pred_sk))
