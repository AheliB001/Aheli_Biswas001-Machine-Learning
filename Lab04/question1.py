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
    y_pred = hypothesis(X, theta)
    return (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

def gradient(X, y, theta):
    m = len(y)
    y_pred = hypothesis(X, theta)
    return (1 / m) * np.dot(X.T, (y_pred - y))

def gradient_descent(X, y, learning_rate, iterations):
    theta = np.zeros(X.shape[1])

    for i in range(iterations):
        grad = gradient(X, y, theta)
        theta = theta - learning_rate * grad

        if i % 100 == 0:
            costt = cost(X, y, theta)
            print(f"Iteration {i}: Cost = {costt}")

    return theta

def main():
    X, y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")

    learning_rate = 0.00001
    iterations = 1000

    theta = gradient_descent(X, y, learning_rate, iterations)

    print("Final learned parameters (theta):")
    print(theta)

if __name__ == "__main__":
    main()