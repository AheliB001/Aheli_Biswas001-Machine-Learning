import pandas as pd
import numpy as np

def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, 0:5].values
    y = df["disease_score_fluct"].values
    return X, y

def hypothesis(X, theta):
    return np.dot(X, theta)

def cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    get_gradient = (1 / m) * np.dot(X.T, (predictions - y))
    return get_gradient

def gradient_descent(X, y, learning_rate, iterations):
    theta = np.zeros(X.shape[1])

    for i in range(iterations):
        get_gradient = gradient(X, y, theta)
        theta = theta - learning_rate * get_gradient

    return theta

def main():
    X, y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")

    X_final = np.hstack((np.ones((X.shape[0], 1)), X))

    learning_rate = 0.0001
    iterations = 100

    theta = gradient_descent(X_final, y, learning_rate, iterations)

    print("Final parameters, theta:")
    print(theta)

if __name__ == "__main__":
    main()


