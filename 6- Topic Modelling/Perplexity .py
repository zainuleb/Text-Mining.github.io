corpus = open('E:\\dataset2.csv').read()
docs = corpus.split('\n')

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(docs)

from sklearn.decomposition import LatentDirichletAllocation
lda1 = LatentDirichletAllocation(n_components = 3)
lda2 = LatentDirichletAllocation(n_components = 2)

lda1.fit(matrix_X[:500])
lda2.fit(matrix_X[:500])

print('lda1: ', lda1.perplexity(matrix_X[500:]))
print('lda2: ', lda2.perplexity(matrix_X[500:]))
