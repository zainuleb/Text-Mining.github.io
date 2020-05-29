corpus = open('E:\\badges.data').read()
docs = corpus.split('\n')
X, y = [], []
for doc in docs:
    y.append(doc[:1])
    X.append(doc[2:])
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(X)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(matrix_X[:200], y[:200])
pred_y = knn.predict(matrix_X[200:])
import numpy as np
true_y = y[200:]
true_y = np.array(true_y)
pred_y = np.array(pred_y)
from sklearn.metrics import precision_score, recall_score, f1_score
print(precision_score(true_y, pred_y, average = 'macro')) # micro, macro, weighted,,,,,,,,, p(l) = tp / tp+fp
print(recall_score(true_y, pred_y, average = 'macro')) # r(l) = tp / tp + fn
print(f1_score(true_y, pred_y, average = 'macro'))
