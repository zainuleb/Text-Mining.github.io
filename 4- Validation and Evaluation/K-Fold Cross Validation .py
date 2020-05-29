corpus = open('E:\\dataset.txt').read()
docs = corpus.split('\n')
X, y = [], []
for doc in docs:
    i, l = doc.split(':')
    X.append(i)
    y.append(l)

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(X)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)

from sklearn.model_selection import KFold
kf = KFold(n_splits = 3)
for id_train, id_test in kf.split(matrix_X):
    print('Train', id_train, 'Test', id_test)
    train_X, test_X = matrix_X[id_train], matrix_X[id_test]
