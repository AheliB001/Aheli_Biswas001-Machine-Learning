import pandas as pd
import numpy as np

def load_data(filename):
    data = pd.read_csv(filename)
    x = data.iloc[:, 0:5].values
    y = data["disease_score_fluct"].values
    return x, y

def hypothesis(x, theta):
    return np.dot(x, theta)

def cost(x, y, theta):
    m = len(y)
    y_pred = hypothesis(x, theta)
    return (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

def gradient(x, y, theta):
    m = len(y)
    y_pred = hypothesis(x, theta)
    return (1 / m) * np.dot(x.T, (y_pred - y))

def main():
    x, y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")

    theta = np.zeros(x.shape[1])
    learning_rate = 0.0001
    iterations = 100

    for i in range(iterations):
        grad = gradient(x, y, theta)
        theta = theta - learning_rate * grad

    print("Final theta values:")
    print(theta)

if __name__ == "__main__":
    main()
