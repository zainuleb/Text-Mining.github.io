corpus = open('E:\\dataset.txt').read()
docs = corpus.split('\n')
X = []
for doc in docs:
    i, l = doc.split(':')
    X.append(i.strip())
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(X)
from sklearn.cluster import AgglomerativeClustering
aggClus = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'cosine')
aggClus.fit(matrix_X.toarray())
print(aggClus.labels_)
