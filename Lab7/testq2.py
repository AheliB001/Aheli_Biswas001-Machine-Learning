import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Load Data
# Assuming 'sonar.csv' is loaded; the last column is the target
sonar = fetch_openml(name='sonar', version=1, as_frame=False)

X = sonar.data
y = sonar.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- METHOD 1: No Pre-processing ---
model_raw = LogisticRegression()
model_raw.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, model_raw.predict(X_test))

# --- METHOD 2: Custom Implementation (Min-Max) ---
def custom_min_max(train, test):
    min_val = train.min(axis=0)
    max_val = train.max(axis=0)
    # Avoid division by zero if max == min
    denom = np.where(max_val - min_val == 0, 1, max_val - min_val)
    train_scaled = (train - min_val) / denom
    test_scaled = (test - min_val) / denom
    return train_scaled, test_scaled

X_train_custom, X_test_custom = custom_min_max(X_train, X_test)
model_custom = LogisticRegression()
model_custom.fit(X_train_custom, y_train)
acc_custom = accuracy_score(y_test, model_custom.predict(X_test_custom))

# --- METHOD 3: Scikit-Learn (StandardScaler) ---
scaler = StandardScaler()
X_train_sklearn = scaler.fit_transform(X_train)
X_test_sklearn = scaler.transform(X_test)
model_sklearn = LogisticRegression()
model_sklearn.fit(X_train_sklearn, y_train)
acc_sklearn = accuracy_score(y_test, model_sklearn.predict(X_test_sklearn))

print(f"Raw Accuracy: {acc_raw:.2%}")
print(f"Custom Scale Accuracy: {acc_custom:.2%}")
print(f"Sklearn Scale Accuracy: {acc_sklearn:.2%}")