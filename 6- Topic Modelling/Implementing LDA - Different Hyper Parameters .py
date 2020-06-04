corpus = ['bread bread bread bread bread bread bread bread bread bread',
         'milk milk milk milk milk milk milk milk milk milk',
         'pet pet pet pet pet pet pet pet pet pet',
         'bread bread bread bread bread bread bread bread bread bread milk milk milk milk milk milk milk milk milk milk']

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(corpus)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 2, topic_word_prior = 0.1, doc_topic_prior = 0.1)
lda.fit(matrix_X)

for topic in lda.components_:
    print([topic[t] for t in topic.argsort()[::-1]])
    
print(lda.transform(matrix_X))
