import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

data=pd.read_csv("sonar_data.csv")
x=data.iloc[:,:-1]

y = data.iloc[:, -1].map({'R': 0, 'M': 1})
model=LogisticRegression()

score = cross_val_score(model, x, y, cv=10, scoring='accuracy') #cross_val_score will fold the dataset into 10 equal folds. so, we do not need to split the data here
print("Accuracy for each fold:", score)
print("Mean Accuracy:", score.mean())
print("Standard Deviation:", score.std())






