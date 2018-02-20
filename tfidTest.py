import pandas as pd
import string
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sa_texts = list(pd.read_csv("spooky_author.csv")["text"])

data = sa_texts[:3000]

#removes punctuations and lowercases all the characters in a word
def clean_word(word):
    def is_letter(l):
        return l in string.ascii_lowercase
    return "".join(list(filter(is_letter, word.lower())))
     
def tokenize(sentence):
    return [clean_word(word) for word in sentence.split()]

#IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
def doc_freqs(docs):
    freqs = {}
    for doc in docs:
        unique_words = set(tokenize(doc))
        for word in unique_words:
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1
    return freqs

d_freqs = doc_freqs(data)

word_to_index = {}
i = 0
for k in d_freqs:
    word_to_index[k] = i
    i += 1

def tfidfize_doc(doc, doc_freqs, num_words):
    #TF(t) = (Number of times term t appears in a document) / (Total Number of terms in the document)
    #IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
    vec = [0]*num_words
    tokens = tokenize(doc)
    token_freqs = {}
    for word in tokens:
        if word in token_freqs:
            token_freqs[word] += 1
        else:
            token_freqs[word] = 1
    for word, v in token_freqs.items():
        tf = v/len(tokens)
        idf = math.log(len(data)/doc_freqs[word])
        vec[word_to_index[word]] = tf*idf
    return vec

vec_reps = list(map(lambda doc: tfidfize_doc(doc, d_freqs, i), data))
sims = cosine_similarity(np.asarray(list(vec_reps)))
list(reversed(sims[5].argsort()))


