#Implement sigmoid function in python and visualize it

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
X=df.iloc[:,0:5].values
Y=df["disease_score_fluct"].values

n=len(Y)
x_matrix=np.column_stack((np.ones(n),X))

def z(theta,x_matrix):
    return np.dot(x_matrix,theta)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def main():
    z = np.linspace(-20,20,100)
    y=sigmoid(z)
    plt.plot(z,y)
    plt.title("Sigmoid function")
    plt.xlabel("z")
    plt.ylabel("sigmoid function")
    plt.show()

main()