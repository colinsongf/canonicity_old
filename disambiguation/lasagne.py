__author__ = 'yutao'

import lasagne
import theano.tensor as T
import pickle
from theano import sparse
import numpy as np
import random
import theano

aff_vocab = pickle.load(open("aff_vocab.pkl", "rb"))
title_vocab = pickle.load(open("title_vocab.pkl", "rb"))
venue_vocab = pickle.load(open("venue_vocab.pkl", "rb"))
attr = pickle.load(open("attr.pkl", "rb"))
authors_map = pickle.load(open("authors_map.pkl", "rb"))
pub_author_map = pickle.load(open("pub_author_map.pkl", "rb"))

aff_idx = dict((c[0], i+1) for i, c in enumerate(aff_vocab))
title_idx = dict((c[0], i+1) for i, c in enumerate(title_vocab))
venue_idx = dict((c[0], i+1) for i, c in enumerate(venue_vocab))

aff_maxlen = 50
title_maxlen = 200
venue_maxlen = 30

embed_dim = 100

g_batch_size = 10
neg_sample = 5

# author - paper pairs
X_ap = []
y_ap = []

# author pairs
X_aa = []
y_aa = []

# def vectorize(data, word_idx, maxlen):
#     X = []
#     for d in data:
#         if not d:
#             x = []
#         else:
#             x = [word_idx[w] for w in d]
#         X.append(x)
#     return pad_sequences(X, maxlen=maxlen)

# aff_data = vectorize([a[1] for a in author_attr], aff_idx, aff_maxlen)
# title_data = vectorize([p[0] for p in pub_attr], title_idx, title_maxlen)
# venue_data = vectorize([p[1] for p in pub_attr], venue_idx, venue_maxlen)


def gen_graph_ap():
    num_ver = len(attr)
    g = []
    gy = []
    for item in pub_author_map:
        g.append(item)
        gy.append(1)
        for k in range(neg_sample):
            g.append((item[0], random.randint(0, num_ver)))
            gy.append(-1)

def gen_graph_aa():
    num_ver = len(attr)
    g = []
    gy = []
    for item in authors_map:
        g.append(item)
        gy.append(1)
        for k in range(neg_sample):
            g.append((item[0], random.randint(0, num_ver)))
            gy.append(-1)
    return gy, gy


aff_var = sparse.csr_matrix('aff', dtype='int')
title_var = sparse.csr_matrix('title', dtype='int')
venue_var = sparse.csr_matrix('venue', dtype='int')
pub_idx = T.vector("pub_idx")
author_idx = T.vector("author_idx")

pairs = T.vector("pairs")
pairs_y = T.vector("pairs_y")

aff_input_layer = lasagne.layers.InputLayer(shape=(None, len(aff_vocab)), input_var=aff_var)
aff_input_layer = lasagne.layers.DenseLayer(aff_input_layer, embed_dim, nonlinearity=lasagne.nonlinearities.softmax)
title_input_layer = lasagne.layers.InputLayer(shape=(None, len(title_vocab)), input_var=title_var)
title_input_layer = lasagne.layers.DenseLayer(title_input_layer, embed_dim, nonlinearity=lasagne.nonlinearities.softmax)
venue_input_layer = lasagne.layers.InputLayer(shape=(None, len(venue_vocab)), input_var=venue_var)
venue_input_layer = lasagne.layers.DenseLayer(venue_input_layer, embed_dim, nonlinearity=lasagne.nonlinearities.softmax)


pairs_input_layer = lasagne.layers.InputLayer(shape=(None, 2), input_var=pairs)
embedding_layer_w = lasagne.layers.SliceLayer(pairs_input_layer, indices=0, axis=1)
embedding_layer_w = lasagne.layers.EmbeddingLayer(embedding_layer_w, input_size=len(attr), output_size=embed_dim)
embedding_layer_c = lasagne.layers.SliceLayer(pairs_input_layer, indices=1, axis=1)
embedding_layer_c = lasagne.layers.EmbeddingLayer(embedding_layer_c, input_size=len(attr), output_size=embed_dim)

embedding_layer_p = lasagne.layers.EmbeddingLayer(pub_idx, input_size=len(attr), output_size=embed_dim, W=embedding_layer_w.W)
embedding_layer_a = lasagne.layers.EmbeddingLayer(author_idx, input_size=len(attr), output_size=embed_dim, W=embedding_layer_w.W)

feature_layer_a = aff_input_layer
feature_layer_a = lasagne.layers.DenseLayer(feature_layer_a, len(attr), nonlinearity=lasagne.nonlinearities.softmax)
feature_layer_p = lasagne.layers.ElemwiseMergeLayer([title_input_layer, venue_input_layer], T.sum)
feature_layer_p = lasagne.layers.DenseLayer(feature_layer_p, len(attr), nonlinearity=lasagne.nonlinearities.softmax)

feature_loss_a = lasagne.objectives.categorical_crossentropy(
        lasagne.layers.get_output(feature_layer_a),
        lasagne.layers.get_output(embedding_layer_p)
)

feature_loss_p = lasagne.objectives.categorical_crossentropy(
    lasagne.layers.get_output(feature_layer_p),
    lasagne.layers.get_output(embedding_layer_p)
)

graph_output_layer = lasagne.layers.ElemwiseMergeLayer([embedding_layer_w, embedding_layer_c], T.mul)
graph_output = lasagne.layers.get_output(graph_output_layer)

graph_loss = - T.log(T.nnet.sigmoid(T.sum(graph_output, axis=1) * pairs_y)).sum()

graph_params = lasagne.layers.get_all_params(graph_output, trainable=True)
graph_updates = lasagne.updates.sgd(graph_loss, graph_params, learning_rate=0.1)

train_graph = theano.function([pairs, pairs_y], graph_loss, updates=graph_updates, on_unused_input="warn")


feature_params_a = lasagne.layers.get_all_params(feature_layer_a)
feature_params_p = lasagne.layers.get_all_params(feature_layer_p)
updates_a = lasagne.updates.sgd(feature_loss_a, feature_params_a, learning_rate=0.1)
updates_p = lasagne.updates.sgd(feature_loss_p, feature_params_p, learning_rate=0.1)

train_feature_a = theano.function([aff_var, author_idx], feature_loss_a, updates=updates_a, on_unused_input="warn")
train_feature_p = theano.function([title_var, venue_var], feature_loss_p, updates=updates_p, on_unused_input="warn")



