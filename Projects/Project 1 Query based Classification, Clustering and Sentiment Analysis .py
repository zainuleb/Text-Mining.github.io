#tokenizing and covertint to lower-case
import nltk.tokenize
raw = open("E:\\dataset.txt").read()
raw = raw.lower()
docs = nltk.tokenize.sent_tokenize(raw)
docs = docs[0].split('\n')

#pre-processing punctuations
from string import punctuation as punc
for d in docs:
    for ch in d:
        if ch in punc:
            d.replace(ch, '')

#removing stopwords, stemming words 
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
ps = PorterStemmer()            
for d in docs:
    for token in nltk.tokenize.word_tokenize(d):
        if token in ENGLISH_STOP_WORDS:
            d.replace(token, '')
        d.replace(token, ps.stem(token))
        
# asking for the test document from the user through direct input
for i in range(len(docs)):
    print('D' + str(i) + ": " + docs[i])
test = input("Enter your text: ")
docs.append(test + ":")

#separating input documents from labels, stripping off the unwanted spaces
X, y = [], []
for d in docs:
    X.append(d[:d.index(":")].lstrip().rstrip())
    y.append(d[d.index(":")+1:].lstrip().rstrip())

#Vectorizing with Tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vec = vectorizer.fit_transform(X)
print(vec.toarray)

#training KNN Classifier
import sklearn
clf = sklearn.neighbors.KNeighborsClassifier(1)
clf.fit(vec[:6], y[:6])
print('Label: ', clf.predict(vec[6]))

#sentiment analysis
from nltk.corpus import wordnet
test_tokens = test.split(' ') 
good = wordnet.synsets('good')
bad = wordnet.synsets('evil')
score_pos = score_neg = 0

for token in test_tokens:
    t = wordnet.synsets(token)
    if len(t) > 0:
        sim_good = wordnet.wup_similarity(good[0], t[0])
        sim_bad = wordnet.wup_similarity(bad[0], t[0])
        if(sim_good is not None) :
            score_pos = score_pos + sim_good
        if(sim_bad is not None):
            score_neg = score_neg + sim_bad

if(score_neg - score_pos > 0.1):
    print('Subjective statement, Negative opinion of strength: %.2f' %score_neg)
elif(score_pos - score_neg > 0.1):
    print('Subjective statement, Positive opinion of strength: %.2f' %score_pos)
else:
    print('Objective statement, No opinion showed')
    
#nearest documents    
nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=2)
nbrs.fit(vec[:6])
closest_docs = nbrs.kneighbors(vec[6])
print('Recommended readings are documents with IDs ', closest_docs[1])
print('having distances ', closest_docs[0])
