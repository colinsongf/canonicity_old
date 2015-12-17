__author__ = 'yutao'

import lasagne
import theano.tensor as T
import pickle
from theano import sparse
import numpy as np
import random
import theano
from collections import defaultdict as dd

aff_vocab = pickle.load(open("aff_vocab.pkl", "rb"))
title_vocab = pickle.load(open("title_vocab.pkl", "rb"))
venue_vocab = pickle.load(open("venue_vocab.pkl", "rb"))
attr = pickle.load(open("attr.pkl", "rb"))
authors_map = pickle.load(open("authors_map.pkl", "rb"))
pub_author_map = pickle.load(open("pub_author_map.pkl", "rb"))

aff_idx = dict((c[0], i+1) for i, c in enumerate(aff_vocab))
title_idx = dict((c[0], i+1) for i, c in enumerate(title_vocab))
venue_idx = dict((c[0], i+1) for i, c in enumerate(venue_vocab))

aff_vocab = dict(aff_vocab)
title_vocab = dict(title_vocab)
venue_vocab = dict(venue_vocab)

aff_maxlen = 50
title_maxlen = 200
venue_maxlen = 30

embed_dim = 100

batch_size = 10
g_batch_size = 10
neg_sample = 5

path_size = 5
window_size = 3

# author - paper pairs
X_ap = []
y_ap = []

# author pairs
X_aa = []
y_aa = []

def vectorize(data, word_idx, vocab):
    x = np.zeros((len(data), max(word_idx.values())+1))
    for i, d in enumerate(data):
        if d:
            for f in d:
                x[i, word_idx[f]] = .1 / vocab[f]
    return x

f_aff = []
f_title = []
f_venue = []
idx_a = []
idx_p = []
for i, d in enumerate(attr):
    if d[0] == "pub":
        f_title.append(d[1])
        f_venue.append(d[2])
        idx_p.append(i)
    elif d[0] == "author":
        f_aff.append(d[2])
        idx_a.append(i)


aff_data = vectorize(f_aff, aff_idx, aff_vocab)
title_data = vectorize(f_title, title_idx, title_vocab)
venue_data = vectorize(f_venue, venue_idx, venue_vocab)

def gen_feature_a():
    while True:
        idx = np.array(np.random.permutation(len(aff_data.shape[0])))
        i = 0
        while i < idx.shape[0]:
            j = min(idx.shape[0], batch_size)
            yield aff_data[idx[i: j]], idx[i: j]
            i = j

def gen_feature_p():
    while True:
        idx = np.array(np.random.permutation(len(aff_data.shape[0])))
        i = 0
        while i < idx.shape[0]:
            j = min(idx.shape[0], batch_size)
            yield title_data[idx[i: j]], venue_data[idx[i: j]], idx[i: j]
            i = j


def gen_graph_ap():
    graph = dd(list)
    num_ver = len(attr)
    for e in pub_author_map:
        graph[e[0]].append(e[1])
        graph[e[1]].append(e[1])

    while True:
        idx = np.random.permutation(num_ver)
        i = 0
        while i < idx.shape[0]:
            g = []
            gy = []
            j = min(idx.shape[0], i+g_batch_size)
            for k in idx[i: j]:
                if len(graph[k]) == 0:
                    continue
                path = [k]
                for _ in range(path_size):
                    path.append(random.choice(graph[path[-1]]))
                for l in range(len(path)):
                    for m in range(l-window_size, l + window_size + 1):
                        if m < 0 or m >= len(path):
                            continue
                        g.append([path[l], path[m]])
                        gy.append(1.0)
                        for k in range(neg_sample):
                            g.append((path[l], random.randint(0, num_ver)))
                            gy.append(-1.0)
            yield np.array(g, dtype=np.int32), np.array(gy, dtype=np.float32)
            i = j

def gen_graph_aa():
    graph = dd(list)
    num_ver = len(attr)
    for e in authors_map:
        graph[e[0]].append(e[1])
        graph[e[1]].append(e[1])

    while True:
        idx = np.random.permutation(num_ver)
        i = 0
        while i < idx.shape[0]:
            g = []
            gy = []
            j = min(idx.shape[0], i+g_batch_size)
            for k in idx[i: j]:
                if len(graph[k]) == 0:
                    continue
                path = [k]
                for _ in range(path_size):
                    path.append(random.choice(graph[path[-1]]))
                for l in range(len(path)):
                    for m in range(l-window_size, l + window_size + 1):
                        if m < 0 or m >= len(path):
                            continue
                        g.append([path[l], path[m]])
                        gy.append(1.0)
                        for k in range(neg_sample):
                            g.append((path[l], random.randint(0, num_ver)))
                            gy.append(-1.0)
            yield np.array(g, dtype=np.int32), np.array(gy, dtype=np.float32)
            i = j


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
train_feature_p = theano.function([title_var, venue_var, pub_idx], feature_loss_p, updates=updates_p, on_unused_input="warn")

iter = 0
while True:
    g1, gy1 = next(gen_graph_aa)
    loss1 = train_graph(g1, gy1)
    g2, gy2 = next(gen_graph_ap)
    loss2 = train_graph(g2, gy2)
    f_a, idx_a = next(gen_feature_a)
    loss3 = train_feature_a(f_a, idx_p)
    f_t, f_v, idx_p = next(gen_feature_p)
    loss4 = train_feature_p(f_t, f_v, idx_p)
    print(iter, loss1, loss2, loss3, loss4)
    iter += 1