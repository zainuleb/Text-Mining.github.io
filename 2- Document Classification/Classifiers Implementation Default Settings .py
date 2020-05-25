#reading data from the file
corpus = open('E:\\dataset.txt').read()

#separating data into documents and labels.
docs = corpus.split('\n')
X, y = [], []
for doc in docs:
    i, l = doc.split(':')
    X.append(i.strip())
    y.append(l.strip())

#Structure input data
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(X)

#Applying K-Nearest Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3, algorithm = 'brute', weights = 'distance')
knn.fit(matrix_X[:5], y[:5])
print('KNN Classifier, Label: ' + str(knn.predict(matrix_X[5])))
print('KNN Classifier, prob.' + str(knn.predict_proba(matrix_X[5])))

#Applying Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
nbc = MultinomialNB(alpha = 0.2, fit_prior = False, class_prior = [0.6, 0.4])
nbc.fit(matrix_X[:5], y[:5])
print('Naive Bayes Classifier, Label: ' + str(nbc.predict(matrix_X[5])))
print('Naive Bayes Classifier, prob.' + str(nbc.predict_proba(matrix_X[5])))

#Applying Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth = 2)
dtc.fit(matrix_X[:5], y[:5])
print('Decision Tree Classifier, Label: ' + str(dtc.predict(matrix_X[5])))
print('Decision Tree Classifier, prob.' + str(dtc.predict_proba(matrix_X[5])))

#Applying Linear Classifier
from sklearn.linear_model import SGDClassifier
lc = SGDClassifier(max_iter = 1000)
lc.fit(matrix_X[:5], y[:5])
print('Linear Classifier, Label: ' + str(lc.predict(matrix_X[5])))
#Linear Classifier doesn't have the probability value as it operates differently
