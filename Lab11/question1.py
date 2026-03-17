import numpy as np
import math
from sklearn.datasets import load_iris

data=load_iris()
x=data.data
y=data.target

def entropy(labels):
    total=len(labels) #total number of samples
    counts=np.bincount(labels) #as many labels so we use bincount
    ent=0
    for c in counts:
        if c>0:
            p=c/total
            ent = ent - p*math.log(p,2)
    return ent

#to find best split
def split(x,y):
    best_gain=-1 #store highest information
    parent_entropy=entropy(y)
    n_features=x.shape[1] #number of features
    for feature in range(n_features):
        values=np.sort(np.unique(x[:,feature]))

        #generate candidate split
        for i in range(len(values)-1):
            threshold=values[i+1]+values[i]/2
            left = y[x[:, feature] <= threshold]
            right = y[x[:, feature] > threshold]

            #split data
            left_entropy = entropy(left)
            right_entropy = entropy(right)

            weighted_entropy = (len(left) / len(y)) * left_entropy + (len(right) / len(y)) * right_entropy
            gain=parent_entropy-weighted_entropy

            if gain>best_gain:
                best_gain=gain
                best_feature=feature
                best_threshold=threshold
    return best_feature, best_threshold
feature,threshold=split(x,y)
print(feature)
print(threshold)