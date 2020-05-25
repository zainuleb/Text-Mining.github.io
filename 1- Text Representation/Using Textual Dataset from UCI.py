corpus = open('E:\\buttons_amazon_kindle.txt.data').read()
docs = corpus.split('\n')
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(docs)
print(X.toarray())
print(vec.get_feature_names())
print(len(vec.get_feature_names()))
print(vec.vocabulary_)
