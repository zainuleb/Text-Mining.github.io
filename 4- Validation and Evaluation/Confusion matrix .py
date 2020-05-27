corpus = open('E:\\badges.data').read()
docs = corpus.split('\n')
X, y = [], []
for doc in docs:
    y.append(doc[:1])
    X.append(doc[2:])
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(X)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 2)
dt.fit(matrix_X[:200], y[:200])
pred_y = dt.predict(matrix_X[200:])
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y[200:], pred_y))
