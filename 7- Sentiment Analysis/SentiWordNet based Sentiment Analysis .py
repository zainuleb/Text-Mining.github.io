from nltk.corpus import wordnet as wnet
from nltk.corpus import sentiwordnet as swnet

review = 'i like to have a nice cup of tea'
review2 = 'i have a very bad experience with this product'
review3 = 'it was ok'
review4 = 'i love the shape but color is bad'
tokens = review4.split(' ')

pos_total = 0
neg_total = 0
for t in tokens:
    syn_t = wnet.synsets(t)
    if len(syn_t) > 0:
        syn_t = syn_t[0]
        senti_syn_t = swnet.senti_synset(syn_t.name())
        if senti_syn_t.pos_score() > senti_syn_t.neg_score():
            pos_total += senti_syn_t.pos_score()
        else:
            neg_total += senti_syn_t.neg_score()

print(pos_total, '-', neg_total)
