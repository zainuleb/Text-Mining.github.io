corpus = ['text text mining is interesting',
          'text mining is the same as data mining',
          'text and data mining have few differences']
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(binary = 'true', max_df = 2, min_df = 2, max_features =2, ngram_range=(1,3))
X = vec.fit_transform(corpus)
print(X.toarray())
print(vec.get_feature_names())
print(len(vec.get_feature_names()))
print(vec.vocabulary_)
