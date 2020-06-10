import nltk
corpus = open('E:\\dataset2.csv').read()
corpus = corpus.lower()
docs = corpus.split('\n')

#creatinig a list of the only POS tags that we are interested in
words = nltk.tokenize.word_tokenize(docs[0])
allowed_tags = ['VBP', 'VB', 'VBG', 'JJ', 'NN', 'RB']

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from string import punctuation as punc

pdocs = []
for d in docs:
    words = nltk.tokenize.word_tokenize(d)
    words_tags = nltk.pos_tag(words)
    for w, t in words_tags:
        if w in punc:
            words.remove(w)
        elif w in ENGLISH_STOP_WORDS:
            words.remove(w)
        elif t not in allowed_tags:
            words.remove(w)
    pd = ' '.join(words)
    pdocs.append(pd)

#Structuring our input documents with CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(pdocs)

#LDA (Topic modeling)
features = vec.get_feature_names()
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 5)
lda.fit(matrix_X)

#Sentiment Analysis
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

for tid, topics in enumerate(lda.components_):
    print('topic ID: ', tid)
    top_words = [features[i] for i in topics.argsort()[:-10:-1]]
    print(top_words)
    score = 0
    for w in top_words:
        senti_synset = swn.senti_synset(wn.synsets(w)[0].name())
        score += senti_synset.pos_score() - senti_synset.neg_score()
    print('Sentiment Score: ', score)
