import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def gradient_descent(X, y, learning_rate=0.01, iterations=2000):
    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * (X.T @ errors)
        theta = theta - learning_rate * gradient

    return theta

def normal_equation(X, y):
    XT = X.T
    theta = np.linalg.inv(XT @ X) @ XT @ y
    return theta

df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

X = df.iloc[:, :5].values
y = df["disease_score_fluct"].values

X = (X - X.mean(axis=0)) / X.std(axis=0)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
X_tr_b = np.c_[np.ones(X_tr.shape[0]), X_tr]
X_te_b = np.c_[np.ones(X_te.shape[0]), X_te]

theta_gd = gradient_descent(X_tr_b, y_tr)
y_pred_gd = X_te_b @ theta_gd

theta_ne = normal_equation(X_tr_b, y_tr)
y_pred_ne = X_te_b @ theta_ne

lr = LinearRegression()
lr.fit(X_tr, y_tr)
y_pred_sk = lr.predict(X_te)

print("SIMULATED DATASET")
print("GD MSE:", mean_squared_error(y_te, y_pred_gd))
print("NE MSE:", mean_squared_error(y_te, y_pred_ne))
print("SK MSE:", mean_squared_error(y_te, y_pred_sk))

df = pd.read_csv("Admission_Predict.csv")
df.columns = df.columns.str.strip()

X = df.drop(columns=["Chance of Admit"]).values
y = df["Chance of Admit"].values

X = (X - X.mean(axis=0)) / X.std(axis=0)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

X_tr_b = np.c_[np.ones(X_tr.shape[0]), X_tr]
X_te_b = np.c_[np.ones(X_te.shape[0]), X_te]

theta_gd = gradient_descent(X_tr_b, y_tr)
y_pred_gd = X_te_b @ theta_gd

theta_ne = normal_equation(X_tr_b, y_tr)
y_pred_ne = X_te_b @ theta_ne

lr = LinearRegression()
lr.fit(X_tr, y_tr)
y_pred_sk = lr.predict(X_te)

print("\nADMISSIONS DATASET")
print("GD R2:", r2_score(y_te, y_pred_gd))
print("NE R2:", r2_score(y_te, y_pred_ne))
print("SK R2:", r2_score(y_te, y_pred_sk))
