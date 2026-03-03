"""from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def main():
    sonar = fetch_openml(name='sonar', version=1, as_frame=False)
   
    X = sonar.data
    y = sonar.target
    # print(X.head())

    k = 10  # K-fold
    samples = len(X)
    idx = np.arange(samples)
    np.random.shuffle(idx)

    test_size = int(samples / k)
    train_size = int(samples - test_size)

    accuracy_list = []
    for fold in range(k):
        start = fold * test_size
        end = start + test_size
        test_set = idx[start:end]
        train_set = np.concatenate((idx[:start], idx[end:]))

        X_test = X[test_set]
        X_train = X[train_set]
        y_train = y[train_set]
        y_test = y[test_set]

        scaler_obj = StandardScaler()
        scaler_parameter = scaler_obj.fit(X_train)
        X_train = scaler_obj.transform(X_train, scaler_parameter)
        X_test = scaler_obj.transform(X_test, scaler_parameter)

        X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

        model = LogisticRegression()
        model.fit(X_train, y_train)
        pred_y = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred_y)
        accuracy_list.append(accuracy)

    print("list of accuracies:",accuracy_list)

    test_accuracy = np.mean(accuracy_list)
    std_dev = np.std(accuracy_list)

    print("Standard Deviation:", std_dev)
    print("Test Accuracy:", test_accuracy)


if __name__ == '__main__':
    main()
"""

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def main():

    #Load SONAR dataset
    sonar = fetch_openml(name='sonar', version=1, as_frame=False)
    X = sonar.data
    y = sonar.target

    #Creating Pipeline (Scaling + Logistic Regression)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    #Defining 10-fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    #Performing Cross Validation
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    #Printing Results
    print("Accuracy of each fold:")
    print(scores)

    print("\nMean Accuracy:", np.mean(scores))
    print("Standard Deviation:", np.std(scores))


if __name__ == "__main__":
    main()