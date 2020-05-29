corpus = open('E:\\badges.data').read()
docs = corpus.split('\n')
X, y = [], []
for doc in docs:
    y.append(doc[:1])
    X.append(doc[2:])
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(X)

from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

kf = KFold(n_splits = 4)
nb = MultinomialNB()
dt = DecisionTreeClassifier(max_depth = 3)

import numpy as np
y = np.array(y)

f1_nb_score = 0; f1_dt_score = 0
for train_ids, test_ids in kf.split(matrix_X):
    train_X, test_X = matrix_X[train_ids], matrix_X[test_ids]
    train_y, test_y = y[train_ids], y[test_ids]
    nb.fit(train_X, train_y)
    dt.fit(train_X, train_y)
    pred_nb_y = nb.predict(test_X)
    pred_dt_y = dt.predict(test_X)
    f1_nb_score += f1_score(test_y, pred_nb_y, average = 'micro')
    f1_dt_score += f1_score(test_y, pred_dt_y, average = 'micro')
    
print(f1_nb_score/4)
print(f1_dt_score/4)
