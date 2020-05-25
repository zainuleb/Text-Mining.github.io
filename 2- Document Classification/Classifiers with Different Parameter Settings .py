
corpus = open('E:\\badges.data').read()
docs = corpus.split('\n')
X, y = [], [] 
for doc in docs:
    l = doc[:1]
    i = doc[2:]
    X.append(i)
    y.append(l)
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
matrix_X = vec.fit_transform(X)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(matrix_X[:290], y[:290])
#predicted labels of the last four documents
print(knn.predict(matrix_X[290:])) 
#prediction probability of the two labels for each of the last four documents
print(knn.predict_proba(matrix_X[290:])) 
