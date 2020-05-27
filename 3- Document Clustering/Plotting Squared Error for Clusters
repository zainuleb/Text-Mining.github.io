corpus = open('E:\\dataset2.csv').read()
docs = corpus.split('\n')
docs.remove(docs[0])
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(docs)
from sklearn.cluster import KMeans

sse = {}
for i in range(1, 20):
    km = KMeans(n_clusters = i)
    km.fit(matrix_X)
    sse[i] = km.inertia_
    
from matplotlib import pyplot as plt
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
