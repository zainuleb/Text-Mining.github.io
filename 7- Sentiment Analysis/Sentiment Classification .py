corpus = open('E:\\Baby.txt').read()
docs = corpus.split('\n')
docs.remove(docs[0])

X, y = [], []
for d in docs:
    X.append(d[11:])
    y.append(d[5:8])

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
matrix_X = vec.fit_transform(X)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(matrix_X[:950], y[:950])

print(clf.predict(matrix_X[950:]))
