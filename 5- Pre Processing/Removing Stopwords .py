corpus = '    This is my duMmy Dataset. It is part of taught course on text mining.   '

#1. to lowercase
corpus = corpus.lower()

#2 remove white spaces
corpus = corpus.strip()

#3 remove punctuations
from string import punctuation as punc
for ch in corpus:
    if ch in punc:
        corpus = corpus.replace(ch, '')
        
print(corpus)  

#4 stop words removal
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
words = corpus.split(' ')
for word in words:
    if word in ENGLISH_STOP_WORDS:
        words.remove(word)
        
print(words)        
