import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score,auc,roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('Heart.csv')

if 'Unnamed: 0' in data.columns: #Remove unnecessary column if present
    data = data.drop('Unnamed: 0', axis=1)
data = data.dropna() #remove data with missing values
X = data.drop('AHD', axis=1)

X = pd.get_dummies(X, drop_first=True) # Convert categorical features into numerical using one-hot encoding

y = data['AHD']
y = y.map({'No': 0, 'Yes': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_prob=model.predict_proba(X_test)[:,1] # Get probability of positive class
#proba give probability of both classes. We use [:,1] to extract the probability of the positive class (class 1), which is required for thresholding and ROC curve.

threshold=float(input('Enter thresholds: ')) #input from user
if threshold < 0 or threshold > 1:
    print("Invalid threshold! Using default = 0.5")
    threshold = 0.5

y_pred=(y_prob>threshold).astype(int) #Convert probabilities into class predictions using threshold
cm=confusion_matrix(y_test,y_pred)
TN, FP, FN, TP = cm.ravel() #converts matrix into a 1D array

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
sensitivity=recall_score(y_test,y_pred)
specificity=TN/(TN+FP)
f1=f1_score(y_test,y_pred)

print('Accuracy:',accuracy
      ,'Precision:',precision
      ,'Sensitivity:',sensitivity)
print('F1 Score:',f1)
print('Confusion Matrix:\n') #to print confusion matrix in 2D
print("            Predicted")
print("               0        1")
print("Actual 0   TN=", cm[0][0], "  FP=", cm[0][1])
print("Actual 1   FN=", cm[1][0], "  TP=", cm[1][1])

fpr,tpr,_ = roc_curve(y_test,y_prob) #plot roc curve._ is for thresholds.different cutoff values used to draw the ROC curve
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--') #diagonal model
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
