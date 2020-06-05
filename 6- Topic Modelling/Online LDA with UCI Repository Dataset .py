corpus = open('E:\\dataset2.csv').read()
docs = corpus.split('\n')

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(docs)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 2, 
        max_iter = 200, learning_offset = 4.0, learning_method = 'online')

step = matrix_X.shape[0]/10
step = int(step)

index = 0
for i in range(10):
    if i == 9:
        lda.partial_fit(matrix_X[index : ])
    else:
        lda.partial_fit(matrix_X[index : index + step])
    index = index + step
    print('\niteration ', i)
    print(lda.components_)
