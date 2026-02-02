import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def gradient_descent(X, y, lrate=0.001, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(iterations):
        theta -= lrate * (1/m) * X.T @ (X @ theta - y)
    return theta

df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X, y = df.iloc[:, :5].values, df["disease_score_fluct"].values

mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

X_tr = np.c_[np.ones(X_tr.shape[0]), X_tr]
X_te = np.c_[np.ones(X_te.shape[0]), X_te]

theta = gradient_descent(X_tr, y_tr, lrate=0.01, iterations=2000)
y_pred_gd = X_te @ theta

lr = LinearRegression().fit(X_tr[:, 1:], y_tr)
y_pred_sk = lr.predict(X_te[:, 1:])

print("Simulated , GD MSE:", mean_squared_error(y_te, y_pred_gd))
print("Simulated , SK MSE:", mean_squared_error(y_te, y_pred_sk))

X, y = fetch_california_housing(return_X_y=True)

X = (X - X.mean(axis=0)) / X.std(axis=0)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

X_tr = np.c_[np.ones(X_tr.shape[0]), X_tr]
X_te = np.c_[np.ones(X_te.shape[0]), X_te]

theta = gradient_descent(X_tr, y_tr, lrate=0.01, iterations=3000)
y_pred_gd = X_te @ theta

lr = LinearRegression().fit(X_tr[:, 1:], y_tr)
y_pred_sk = lr.predict(X_te[:, 1:])

print("California , GD R2:", r2_score(y_te, y_pred_gd))
print("California , SK R2:", r2_score(y_te, y_pred_sk))



