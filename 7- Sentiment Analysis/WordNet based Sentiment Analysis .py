from nltk.corpus import wordnet
good = wordnet.synsets('good')[0]
bad = wordnet.synsets('bad')[0]

review = 'i like to have a nice cup of tea'
tokens = review.split(' ')

total_score = 0
for t in tokens:
    syn_t = wordnet.synsets(t)
    if len(syn_t) > 0:
        syn_t = syn_t[0]
        score = wordnet.wup_similarity(good, syn_t) - wordnet.wup_similarity(bad, syn_t)
        total_score += score
print(total_score)
