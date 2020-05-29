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

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

for train_ids, test_ids in loo.split(matrix_X):
    print('Train', train_ids, 'Test', test_ids)
    train_X, test_X = matrix_X[train_ids], matrix_X[test_ids]
