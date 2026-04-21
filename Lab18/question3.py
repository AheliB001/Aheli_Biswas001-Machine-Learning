import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("Tweets.csv")

X = df['text']
y = df['airline_sentiment']

#Convert labels -> numbers
y = y.map({'negative': 0, 'neutral': 1, 'positive': 2})

#Convert text -> TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

linear = SVC(kernel='linear').fit(X_train, y_train)
rbf = SVC(kernel='rbf').fit(X_train, y_train)
poly = SVC(kernel='poly', degree=2).fit(X_train, y_train)

print("Linear:", accuracy_score(y_test, linear.predict(X_test)))
print("RBF   :", accuracy_score(y_test, rbf.predict(X_test)))
print("Poly  :", accuracy_score(y_test, poly.predict(X_test)))