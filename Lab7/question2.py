import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("sonar_data.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1].map({'R': 0, 'M': 1})

kf = KFold(n_splits=10, shuffle=True, random_state=42)

model = LogisticRegression(max_iter=1000)

#without normalization
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("Without Normalization:")
print("Mean Accuracy:", scores.mean())

#with normalization
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)


X_std = (X - mean) / std


scores_std = cross_val_score(model, X_std, y, cv=kf, scoring='accuracy')
print("With Manual Normalization:")
print("Mean Accuracy:", scores_std.mean())


#standard scaler
scaler = StandardScaler()
X_sklearn = scaler.fit_transform(X)

scores_sklearn = cross_val_score(model, X_sklearn, y, cv=kf, scoring='accuracy')
print("With Sklearn StandardScaler:")
print("Mean Accuracy:", scores_sklearn.mean())

