corpus = open('E:\\dataset.txt').read()
docs = corpus.split('\n')
input_X, y = [], []
for doc in docs:
    i, l = doc.split(':')
    input_X.append(i.strip())
    y.append(l.strip())
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(binary = 'true', max_df = 2, min_df = 2, max_features =2, ngram_range=(1,3))
X = vec.fit_transform(input_X)
print(X.toarray())
print(vec.get_feature_names())
print(len(vec.get_feature_names()))
print(vec.vocabulary_)
