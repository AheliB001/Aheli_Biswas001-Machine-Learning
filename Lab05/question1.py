import pandas as pd
import numpy as np
import random

def load_data(filename):
    data = pd.read_csv(filename)
    x = data.iloc[:, 0:5].values
    y = data["disease_score_fluct"].values
    return x, y

def hypothesis(x, theta):
    return np.dot(x, theta)

def main():
    x, y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")

    theta = np.zeros(x.shape[1])
    learning_rate = 0.0001
    iterations = 1000
    m = len(y)

    for it in range(iterations):
        #picking 1 random data-point
        i = random.randint(0, m - 1)

        xi = x[i]
        yi = y[i]

        #prediction for that single sample
        y_pred = hypothesis(xi, theta)

        #gradient
        grad = (y_pred - yi) * xi

        #updating parameter
        theta = theta - learning_rate * grad

    print("Final theta values:")
    print(theta)

if __name__ == "__main__":
    main()
