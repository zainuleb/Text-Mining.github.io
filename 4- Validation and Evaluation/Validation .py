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

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(matrix_X, y, train_size = 0.7, shuffle = True)

knn.fit(train_X, train_y)
print(knn.predict(test_X))
print(test_y)
