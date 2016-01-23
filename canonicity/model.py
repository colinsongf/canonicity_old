import logging
import random

import lasagne
import numpy as np
import theano
import theano.tensor as T
from theano import sparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import pickle
from collections import defaultdict as dd

data = pickle.load(open("../data/person_pub_data.pkl", "rb"))
sorted_names = pickle.load(open("sorted_names.pkl", "rb"))
name_to_idx = pickle.load(open("name_to_idx.pkl", "rb"))
vocab = pickle.load(open("vocab.pkl", "rb"))
token_to_idx = pickle.load(open("token_to_idx.pkl", "rb"))
idx_to_token = pickle.load(open("idx_to_token.pkl", "rb"))
fvectors = pickle.load(open("fvectors.pkl", "rb"))

class Canonicity:
    def __init__(self, args):
        self.schema = args.schema
        self.num_nodes = self.schema["num_nodes"]
        self.embedding_dim = args.embedding_dim
        self.learning_rate = args.learning_rate
        self.neg_rate = args.neg_rate
        self.embedding = None
        self.graph_fn = None
        self.attr_fn = {}
        self.anchor_fn = {}
        self.g_learning_rate = 0.1
        self.anchors = []
        self.features = None
        np.random.seed(1)
        random.seed(1)

    def build(self):
        # local graph context
        g_sym = T.imatrix('g')  # a pair of node index (an edge)
        gy_sym = T.vector('gy')  # label of a pair (indicating whether it is a false edge)
        l_g_in = lasagne.layers.InputLayer(shape=(None, 2), input_var=g_sym)
        l_gy_in = lasagne.layers.InputLayer(shape=(None,), input_var=gy_sym)
        # embedding of node i (pivot node)
        l_emb_local_i = lasagne.layers.SliceLayer(l_g_in, indices=0, axis=1)
        l_emb_local_i = lasagne.layers.EmbeddingLayer(l_emb_local_i, input_size=self.num_nodes,
                                                      output_size=self.embedding_dim)
        # embedding of node j (context node)
        l_emb_local_j = lasagne.layers.SliceLayer(l_g_in, indices=1, axis=1)
        l_emb_local_j = lasagne.layers.EmbeddingLayer(l_emb_local_j, input_size=self.num_nodes,
                                                      output_size=self.embedding_dim)
        l_gy = lasagne.layers.ElemwiseMergeLayer([l_emb_local_i, l_emb_local_j], T.mul)
        pgy_sym = lasagne.layers.get_output(l_gy)
        g_loss = -T.log(T.nnet.sigmoid(T.sum(pgy_sym, axis=1) * gy_sym)).sum()
        g_params = lasagne.layers.get_all_params(l_gy, trainable=True)
        g_updates = lasagne.updates.sgd(g_loss, g_params, learning_rate=self.g_learning_rate)
        self.graph_fn = theano.function([g_sym, gy_sym], g_loss, updates=g_updates, on_unused_input='warn')

        self.embedding = l_emb_local_i.W

        # local attributes
        ind_sym = T.ivector('ind')
        l_ind_in = lasagne.layers.InputLayer(shape=(None,), input_var=ind_sym)
        # embedding of current node
        l_emb_f = lasagne.layers.EmbeddingLayer(l_ind_in, input_size=self.num_nodes,
                                                output_size=self.embedding_dim, W=self.embedding)
        x_sym = {}
        y_sym = T.imatrix('y')
        l_x_in = {}
        l_x_hid = {}
        attr_loss = {}
        for n in self.schema["nodes"]:
            x_sym[n] = sparse.csr_matrix(n, dtype='float32')
            l_x_in[n] = lasagne.layers.InputLayer(shape=(None, self.schema["shape"][n]), input_var=x_sym[n])
            l_x_hid[n] = lasagne.layers.DenseLayer(l_x_in[n], self.embedding_dim)
            l_ay = lasagne.layers.ElemwiseMergeLayer([l_x_hid, l_emb_f], T.mul)
            pay_sym = lasagne.layers.get_output(l_ay)
            attr_loss[n] = -T.log(T.nnet.sigmoid(T.sum(pay_sym, axis=1) * y_sym)).sum()
            attr_params = lasagne.layers.get_all_params(l_ay, trainable=True)
            attr_updates = lasagne.updates.sgd(attr_loss[n], attr_params, learning_rate=self.g_learning_rate)
            self.attr_fn[n] = theano.function([x_sym[n], y_sym, ind_sym], attr_loss[n], updates=attr_updates, on_unused_input='warn')

        # alignment
        anchor_sym = T.imatrix('anchor')
        anchor_y_sym = T.vector('anchor_y')
        l_a_in = lasagne.layers.InputLayer(shape=(None, 2), input_var=anchor_sym)
        l_emb_anchor_i = lasagne.layers.SliceLayer(l_a_in, indices=0, axis=1)
        l_emb_anchor_i = lasagne.layers.EmbeddingLayer(l_emb_anchor_i, input_size=self.num_nodes,
                                                       output_size=self.embedding_dim, W=self.embedding)
        l_emb_anchor_j = lasagne.layers.SliceLayer(l_a_in, indices=1, axis=1)
        l_emb_anchor_j = lasagne.layers.EmbeddingLayer(l_emb_anchor_j, input_size=self.num_nodes,
                                                       output_size=self.embedding_dim, W=self.embedding)
        l_anchor_y = lasagne.layers.ElemwiseMergeLayer([l_emb_anchor_i, l_emb_anchor_j], T.mul)
        p_anchor_y_sym = lasagne.layers.get_output(l_anchor_y)
        anchor_loss = -T.log(T.nnet.sigmoid(T.sum(p_anchor_y_sym, axis=1) * anchor_y_sym)).sum()
        anchor_params = lasagne.layers.get_all_params(l_anchor_y, trainable=True)
        anchor_updates = lasagne.updates.sgd(anchor_loss, anchor_params, learning_rate=self.g_learning_rate)
        self.anchor_fn = theano.function([anchor_sym, anchor_y_sym], g_loss, updates=anchor_updates, on_unused_input='warn')

        # self.x_sym, self.y_sym, self.ind_sym = x_sym, y_sym, ind_sym

    def get_feature(self, n):
        x, y, ind = [], [], []
        x.append(self.features[n])
        y.append(1.0)
        ind.append(n)
        for _ in range(self.neg_rate):
            m = random.randint(self.num_nodes)
            x.append(self.features[n])
            y.append(-1.0)
            ind.append(m)
        return x, y, ind

    def gen_context_graph(self):
        while True:
            idx = np.random.permutation(self.num_nodes)
            for i in range(idx.shape[0]):
                g, gy = [], []
                pivot_node = data[i]
                clique = []
                if "a" in pivot_node:
                    clique.append((pivot_node["i"], "p"))
                    for a in pivot_node["a"]:
                        clique.append((a["i"], "a"))
                else:
                    clique.append((pivot_node["i"], "a"))
                    p = data[pivot_node["p"]]
                    clique.append((p["i"], "p"))
                    for a in p["a"]:
                        if a["i"] != pivot_node["i"]:
                            clique.append((a["i"], "a"))
                x, y, ind = self.get_feature(clique[0])
                for n in clique[1:]:
                    x_, y_, ind_ = self.get_feature(n)
                    x += x_
                    y += y_
                    ind += ind_
                    g.append((clique[0][0], n[0]))
                    gy.append(1.0)
                    for _ in range(self.neg_rate):
                        g.append((clique[0][0], random.randint(self.num_nodes)))
                        gy.append(-1.0)
                yield (np.array(g, dtype=np.int32),
                       np.array(gy, dtype=np.int32),
                       sparse.vstack(x),
                       np.array(y, dtype=np.int32),
                       np.array(ind, dtype=np.int32))



    def gen_anchor(self):
        while True:
            idx = np.random.permutation(len(self.anchors))
            for i in range(idx.shape[0]):
                anchor, anchor_y = [], []
                anchor.append(self.anchors[i])
                anchor_y.append(1.0)
                for _ in range(self.neg_rate):
                    anchor.append((self.anchors[i][0], random.randint(self.num_nodes)))
                    anchor_y.append(-1.0)
                yield np.array(anchor, dtype=np.int32), np.array(anchor_y, dtype=np.int32)

    def train(self):
        print('start training')

        max_acc = 0.0

        while True:


def sample_anchor(num_names, num_node, neg_rate):
    current_name = random.randint(0, num_names - 1)
    name = sorted_names[current_name]
    pubs = name_to_idx[name]
    if len(pubs) < 2:
        return []
    true_pair = tuple([data[k[0]]["a"][k[1]]["i"] for k in random.sample(name_to_idx[name], 2)])
    samples = [(true_pair, 1)]
    for i in [(0, 1), (1, 0)]:
        for j in range(neg_rate):
            idx = random.randint(0, num_node - 1)
            false_pair = [-1, -1]
            false_pair[i[0]] = true_pair[i[0]]
            if "a" in data[idx]:
                if len(data[idx]["a"]) == 0:
                    continue
                false_pair[i[1]] = random.sample(data[idx]["a"], 1)[0]["i"]
            else:
                false_pair[i[1]] = idx
            samples.append((tuple(false_pair), 0))
    return samples


def get_path(node, neg_rate, num_vertex):
    pathes = []
    pub = data[data[node]["p"]]
    for a in pub["a"]:
        if not a["i"] == data[node]["i"]:
            path = ((node, fvectors['n'][node], fvectors['o'][node]),
                    (pub["i"], fvectors['t'][pub['i']], fvectors['k'][pub['i']], fvectors['v'][pub['i']]),
                    (a['i'], fvectors['n'][a['i']], fvectors['o'][a['i']]), 1, 1)
            pathes.append(path)
    for i in range(neg_rate):
        idx = random.randint(0, num_vertex - 1)
        if "a" in data[idx]:
            if len(data[idx]["a"]) == 0:
                continue
            a_idx = random.randint(0, len(data[idx]["a"]) - 1)
            p_id = data[idx]["i"]
            a_id = data[idx]["a"][a_idx]["i"]
            path = ((node, fvectors['n'][node], fvectors["o"][node]),
                    (p_id, fvectors["t"][p_id], fvectors["k"][p_id], fvectors["v"][p_id]),
                    (a_id, fvectors["n"][a_id], fvectors["o"][a_id]),
                    0, 0)
            pathes.append(path)
        else:
            p_id = pub["i"]
            a_id = data[idx]["i"]
            path = ((node, fvectors['n'][node], fvectors["o"][node]),
                    (p_id, fvectors["t"][p_id], fvectors["k"][p_id], fvectors["v"][p_id]),
                    (a_id, fvectors["n"][a_id], fvectors["o"][a_id]),
                    1, 0)
            pathes.append(path)
    return pathes


def get_context_pair(pair):
    pathes = []
    for i, leg in enumerate(["left", "right"]):
        pathes.append(get_path(pair[i], 10, 100))
    path_pairs = []
    for p1 in pathes[0]:
        for p2 in pathes[1]:
            path_pairs.append((p1, p2))
    return path_pairs


def gen_batch():
    num_names = 800000
    num_node = len(data)
    neg_rate = 10
    batch_size = 1
    while True:
        instances = dd(list)
        for _ in range(batch_size):
            samples = sample_pair(num_names, num_node, neg_rate)
            if len(samples) == 0:
                continue
            for s in samples:
                path_pairs = get_context_pair(s[0])
                legs = ["left", "right"]
                for p in path_pairs:
                    instances["output_alignment"].append(s[1])
                    for i, n in enumerate(p):
                        # outputs
                        instances["%s_output_edge_1" % legs[i]].append(n[3])
                        instances["%s_output_edge_2" % legs[i]].append(n[4])

                        # vertex index
                        for k1 in range(3):
                            instances["%s_vertex_%s" % (legs[i], k1)].append(np.array(n[k1][0]))
                        # attribute feature vector
                        for k1 in range(3):
                            for k2 in range(1, len(n[k1])):
                                instances["%s_input_attr_%s_%s" % (legs[i], k1, k2 - 1)].append(n[k1][k2])
                                instances["%s_output_vertex_%s_%s" % (legs[i], k1, k2 - 1)].append(1)
        for k in instances:
            instances[k] = sparse.vstack(instances[k]).toarray()
        yield instances
