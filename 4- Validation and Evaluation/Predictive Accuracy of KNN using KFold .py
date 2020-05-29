corpus = open('E:\\dataset.txt').read()
docs = corpus.split('\n')
X, y = [], []
for doc in docs:
    i, l = doc.split(':')
    X.append(i.strip())
    y.append(l.strip())
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(X)

from sklearn.model_selection import KFold
kf = KFold(n_splits = 3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)

import numpy as np
y = np.array(y)

from sklearn.metrics import accuracy_score

score = 0
for train_ids, test_ids in kf.split(matrix_X):
    train_X, test_X = matrix_X[train_ids], matrix_X[test_ids]
    train_y, test_y = y[train_ids], y[test_ids]
    knn.fit(train_X, train_y)
    pred_y = knn.predict(test_X)
    score += accuracy_score(test_y, pred_y, normalize = True)
print(score/3)
