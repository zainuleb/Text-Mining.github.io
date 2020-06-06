corpus = open('E:\\dataset2.csv').read()
docs = corpus.split('\n')

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
matrix_X = vec.fit_transform(docs)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 6)
lda.fit(matrix_X)

features = vec.get_feature_names()

for tid, topic in enumerate(lda.components_):
    print('topic: ', tid)
    print('wordID: ', topic.argsort()[::-1])
    print('word: ', [features[i] for i in topic.argsort()[:-10:-1]])
    print('prob: ', [topic[i] for i in topic.argsort()[:-10:-1]])
