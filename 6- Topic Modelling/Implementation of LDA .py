corpus = ['bread bread bread bread bread bread bread bread bread bread',
         'milk milk milk milk milk milk milk milk milk milk',
         'pet pet pet pet pet pet pet pet pet pet',
         'bread bread bread bread bread bread bread bread bread bread milk milk milk milk milk milk milk milk milk milk']

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(corpus)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 2)
lda.fit(matrix_X)

print(lda.components_)

features = vec.get_feature_names()

for tid, topic in enumerate(lda.components_):
    print('topic ID: ', tid)
    print('word IDs: ', topic.argsort()[::-1])
    print('words: ', [features[i] for i in topic.argsort()[::-1]])
    print('prob: ', [topic[i] for i in  topic.argsort()[::-1]])
