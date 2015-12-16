__author__ = 'yutao'

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Merge, LambdaMerge, Lambda
from keras.models import Graph
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
import pickle
from theano import tensor as T
import random

aff_vocab = pickle.load(open("aff_vocab.pkl", "rb"))
title_vocab = pickle.load(open("title_vocab.pkl", "rb"))
venue_vocab = pickle.load(open("venue_vocab.pkl", "rb"))
author_attr = pickle.load(open("author_attr.pkl", "rb"))
pub_attr = pickle.load(open("pub_attr.pkl", "rb"))
authors_map = pickle.load(open("authors_map.pkl", "rb"))
pub_author_map = pickle.load(open("pub_author_map.pkl", "rb"))

aff_idx = dict((c[0], i+1) for i, c in enumerate(aff_vocab))
title_idx = dict((c[0], i+1) for i, c in enumerate(title_vocab))
venue_idx = dict((c[0], i+1) for i, c in enumerate(venue_vocab))

aff_maxlen = 50
title_maxlen = 200
venue_maxlen = 30

embed_dim = 100

neg_sample = 5

# author - paper pairs
X_ap = []
y_ap = []

# author pairs
X_aa = []
y_aa = []

def gen_pub_author_map_inst():
    pairs = []
    y = []
    for item in pub_author_map:
        pairs.append(item)
        y.append(1)
        for k in range(neg_sample):
            pairs.append((item[0], random.randint(0, len(pub_attr))))
            y.append(0)
    return pairs, y

def gen_authors_map_inst():
    pairs = []
    y = []
    for item in authors_map:
        pairs.append(item)
        y.append(1)
        for k in range(neg_sample):
            pairs.append((item[0], random.randint(0, len(author_attr))))
            y.append(0)
    return pairs, y

def vectorize(data, word_idx, maxlen):
    X = []
    for d in data:
        if not d:
            x = []
        else:
            x = [word_idx[w] for w in d]
        X.append(x)
    return pad_sequences(X, maxlen=maxlen)

aff_data = vectorize([a[1] for a in author_attr], aff_idx, aff_maxlen)
title_data = vectorize([p[0] for p in pub_attr], title_idx, title_maxlen)
venue_data = vectorize([p[1] for p in pub_attr], venue_idx, venue_maxlen)


# embed authors
aff_encoder = Sequential()
aff_encoder.add(Embedding(input_dim=len(aff_vocab)+1,
                          output_dim=embed_dim,
                          input_length=aff_maxlen))

title_encoder = Sequential()
title_encoder.add(Embedding(input_dim=len(aff_vocab)+1,
                            output_dim=embed_dim,
                            input_length=title_maxlen))

venue_encoder = Sequential()
title_encoder.add(Embedding(input_dim=len(aff_vocab)+1,
                            output_dim=embed_dim,
                            input_length=venue_maxlen))

pub_emb = Sequential()
pub_emb.add(Embedding(input_dim=len(pub_attr),
                      output_dim=embed_dim,
                      input_length=2))

author_emb = Sequential()
author_emb.add(Embedding(input_dim=len(author_attr),
                         output_dim=embed_dim,
                         input_length=2))

#
# def pub_author_edge_func(inputs):
#     X_a = inputs[0]
#     X_p = inputs[1]
#     dot = T.sum(X_a * X_p, axis=1)
#     dot = T.reshape(dot, (X_a.shape[0], 1))
#     return dot
#
# pub_author_edge = Sequential()
# pub_author_edge.add(LambdaMerge([pub, author], function=pub_author_edge_func))
#
# def author_match_func(inputs):
#     return None
#
#
# author.add(Lambda(function=author_match_func))
#
# def sum_func(inputs):
#     return inputs[0] + inputs[1]



model = Graph()
model.add_input(name="aff_data", input_shape=(aff_maxlen,), dtype=int)
model.add_input(name="title_data", input_shape=(title_maxlen,), dtype=int)
model.add_input(name="venue_data", input_shape=(venue_maxlen,), dtype=int)
model.add_input(name="pub_author_map", input_shape=(2,), dtype=int)
model.add_input(name="authors_map", input_shape=(2,), dtype=int)

model.add_node(Embedding(input_dim=len(aff_vocab)+1,
                         output_dim=embed_dim,
                         input_length=aff_maxlen),
               name='aff_embedding', input='aff_data')
model.add_node(Embedding(input_dim=len(title_vocab)+1,
                         output_dim=embed_dim,
                         input_length=title_maxlen),
               name='title_embedding', input='title_data')
model.add_node(Embedding(input_dim=len(venue_vocab)+1,
                         output_dim=embed_dim,
                         input_length=venue_maxlen),
               name='venue_embedding', input='venue_data')

model.add_node(Embedding(input_dim=len(pub_attr),
                         output_dim=embed_dim,
                         input_length=2,
                         initial_weight=),
               name="pub_embedding", input="pub_author_map")

author_emb = Sequential()
author_emb.add(Embedding(input_dim=len(author_attr),
                         output_dim=embed_dim,
                         input_length=2))

model.add_node(Lambda(function=sum_func), inputs=["title_data", "venue_data"])
model.add_node(Lambda(function=pub_author_edge), inputs=[""])



