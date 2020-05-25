corpus = open('E:\\dataset-CalheirosMoroRita-2017.csv').read()
docs = corpus.split('\n')

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(docs)

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
aggClus = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'manhattan')
aggClus.fit(matrix_X.toarray())
print(aggClus.labels_)

kmeans = KMeans(n_clusters = 4)
kmeans.fit(matrix_X[:300])
print(kmeans.predict(matrix_X[300:]))
