import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load data
data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MSE function
def mse(values):
    mean = np.mean(values)
    return np.mean((values - mean) ** 2)

# Find best split
def split(X, y):

    best_error = float('inf')
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i+1]) / 2

            left = y[X[:, feature] <= threshold]
            right = y[X[:, feature] > threshold]

            if len(left) == 0 or len(right) == 0:
                continue

            error = (len(left)/len(y))*mse(left) + (len(right)/len(y))*mse(right)

            if error < best_error:
                best_error = error
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

# Build tree
def build_tree(X, y, depth=0, max_depth=3):

    if depth >= max_depth or len(y) <= 2:
        return {"value": np.mean(y)}

    feature, threshold = split(X, y)

    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold

    return {
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(X[left_idx], y[left_idx], depth+1, max_depth),
        "right": build_tree(X[right_idx], y[right_idx], depth+1, max_depth)
    }

# Predict single sample
def predict_sample(tree, x):

    if "value" in tree:
        return tree["value"]

    if x[tree["feature"]] <= tree["threshold"]:
        return predict_sample(tree["left"], x)
    else:
        return predict_sample(tree["right"], x)

# Predict all
def predict(tree, X):
    return np.array([predict_sample(tree, x) for x in X])

tree = build_tree(X_train, y_train)
prediction = predict(tree, X_test)
mse_value = np.mean((y_test - prediction) ** 2)

print("MSE =", mse_value)