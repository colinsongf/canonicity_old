import pickle
import random
from collections import defaultdict as dd

from keras.layers.core import Dense, Layer, Lambda,Activation
from keras.layers.core import initializations, activations
from keras.layers.embeddings import Embedding
from keras.models import Graph, Sequential
from theano import tensor as T
from keras import backend as K
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

data = pickle.load(open("../data/person_pub_data.pkl", "rb"))
sorted_names = pickle.load(open("sorted_names.pkl", "rb"))
name_to_idx = pickle.load(open("name_to_idx.pkl", "rb"))

title_df = dd(int)
venue_df = dd(int)
aff_df = dd(int)


class NodeContextProduct(Layer):
    '''
        This layer turns a pair of words (a pivot word + a context word,
        ie. a word from the same context, or a random, out-of-context word),
        indentified by their index in a vocabulary, into two dense reprensentations
        (word representation and context representation).
        Then it returns activation(dot(pivot_embedding, context_embedding)),
        which can be trained to encode the probability
        of finding the context word in the context of the pivot word
        (or reciprocally depending on your training procedure).
        The layer ingests integer tensors of shape:
        (nb_samples, 2)
        and outputs a float tensor of shape
        (nb_samples, 1)
        The 2nd dimension encodes (pivot, context).
        input_dim is the size of the vocabulary.
        For more context, see Mikolov et al.:
            Efficient Estimation of Word reprensentations in Vector Space
            http://arxiv.org/pdf/1301.3781v3.pdf
    '''

    def __init__(self, input_dim, proj_dim=128,
                 init='uniform', activation='sigmoid', weights=None):
        super(NodeContextProduct, self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.input = T.imatrix()
        # two different embeddings for pivot word and its context
        # because p(w|c) != p(c|w)
        self.W_w = self.init((input_dim, proj_dim))
        self.W_c = self.init((input_dim, proj_dim))

        self.params = [self.W_w, self.W_c]

        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train=False):
        X = self.get_input(train)
        w = self.W_w[X[:, 0]]  # nb_samples, proj_dim
        c = self.W_c[X[:, 1]]  # nb_samples, proj_dim

        dot = T.sum(w * c, axis=1)
        dot = T.reshape(dot, (X.shape[0], 1))
        return self.activation(dot)

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "proj_dim": self.proj_dim,
            "init": self.init.__name__,
            "activation": self.activation.__name__}

class IntInput(Layer):
    def build(self):
        super(IntInput, self).build()
        self.input = K.placeholder(self._input_shape, dtype='int32')

"""
num_node: number of entity node to be embedded
node_attr_mapping: attributes of each node type
attr_shape: original dimension of each attribute
embedding_dim: dimension of the embedding space
"""
def build_network(num_node, path_type, node_attr_mapping, attr_shape, embedding_dim=100):
    node_type_idx = {}
    attr_type_idx = {}
    num_node_type = 0
    num_attr_type = 0
    for n in node_attr_mapping:
        node_type_idx[n] = num_node_type
        num_node_type += 1
        for a in node_attr_mapping[n]:
            attr_type_idx[a] = num_attr_type
            num_attr_type += 1

    output_loss = []


    model = Graph()

    for i, n in enumerate(path_type):
        for leg in ["left", "right"]:
            model.add_input(name=leg + '_vertex_' + str(i), input_shape=(1,), dtype='int32')
    # shared embedding
    model.add_shared_node(layer=Embedding(input_dim=num_node, output_dim=embedding_dim, input_length=1),
                          name="embedding_pivot",
                          inputs=[(leg + '_vertex_0') for leg in ('left', 'right')],
                          outputs=[(leg + '_embeding_vertex_0') for leg in ('left', 'right')])
    model.add_shared_node(layer=Embedding(input_dim=num_node, output_dim=embedding_dim, input_length=1),
                          name="embedding_context",
                          inputs=[(leg + '_vertex_' + str(i)) for leg in ('left', 'right') for i in range(1, len(path_type))],
                          outputs=[(leg + '_embeding_vertex_' + str(i)) for leg in ('left', 'right') for i in range(1, len(path_type))])

    for i, n in enumerate(path_type):
        attr_layers = []
        for j, a in enumerate(node_attr_mapping[n]):
            for leg in ["left", "right"]:
                model.add_input(name="%s_input_attr_%s_%s" % (leg, i, j), input_shape=(attr_shape[a], ))
            model.add_shared_node(layer=Dense(output_dim=embedding_dim),
                                  name="encoder_attr_%s_%s" % (i, j),
                                  inputs=["%s_input_attr_%s_%s" % (leg, i, j) for leg in ["left", "right"]],
                                  outputs=["%s_encoded_attr_%s_%s" % (leg, i, j) for leg in ["left", "right"]])
            attr_layers.append("_encoded_attr_%s_%s" % (i, j))
            for leg in ["left", "right"]:
                model.add_output("%s_output_vertex_%s_%s" % (leg, i, j),
                                 inputs=["%s_encoded_attr_%s_%s" % (leg, i, j), "%s_embeding_vertex_%s" % (leg, i)],
                                 merge_mode="dot")
                output_loss.append(("%s_output_vertex_%s_%s" % (leg, i, j), "binary_crossentropy"))

        # add vertex attributes loss
    # for i, n in enumerate(path_type):
    #     for leg in ["left", "right"]:
    #         model.add_node(Dense(output_dim=embedding_dim), name="%s_embedding_attr_%s" % (leg, i),
    #                        inputs=[leg+layer_name for layer_name in attr_layers], merge_mode="sum")
    #         model.add_output("%s_output_vertex_%s" % (leg, i),
    #                          inputs=["%s_embedding_attr_%s" % (leg, i), "%s_embeding_vertex_%s" % (leg, i)],
    #                          merge_mode="dot")
    #         output_loss.append(("%s_output_vertex_%s" % (leg, i), "binary_crossentropy"))

    # add local context loss
    for leg in ["left", "right"]:
        for i in range(1, len(path_type)):
            model.add_output("%s_output_edge_%s" % (leg, i),
                             inputs=["%s_embeding_vertex_%s" % (leg, 0), "%s_embeding_vertex_%s" % (leg, i)],
                             merge_mode="dot")
            output_loss.append(("%s_output_edge_%s" % (leg, i), "binary_crossentropy"))

    # add alignment loss
    model.add_output("output_alignment",
                     inputs=["left_embeding_vertex_0", "right_embeding_vertex_0"],
                     merge_mode="dot")
    output_loss.append(("output_alignment", "binary_crossentropy"))

    model.compile('sgd', loss=dict(output_loss))
    return model


vocab = pickle.load(open("vocab.pkl", "rb"))
token_to_idx = pickle.load(open("token_to_idx.pkl", "rb"))
idx_to_token = pickle.load(open("idx_to_token.pkl", "rb"))
fvectors = pickle.load(open("fvectors.pkl", "rb"))


def sample_pair(num_names, num_node, neg_rate):
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
    # print(node)
    # print(data[node]["p"])
    pub = data[data[node]["p"]]
    for a in pub["a"]:
        if not a["i"] == data[node]["i"]:
            path = ((node, fvectors['n'][node], fvectors['o'][node]),
                    (pub["i"], fvectors['t'][pub['i']], fvectors['k'][pub['i']], fvectors['v'][pub['i']]),
                    (a['i'], fvectors['n'][a['i']], fvectors['o'][a['i']]), 1, 1)
            # path = ((node, data[node]['n'], data[node]["o"]),
            #         (pub["i"], pub["t"], pub["k"], pub["v"]),
            #         (a["i"], a["n"], a["o"]), 1, 1)
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
            # path = ((node, data[node]['n'], data[node]["o"]),
            #         (data[idx]["i"], data[idx]["t"], data[idx]["k"], data[idx]["v"]),
            #         (data[idx]["a"][a_idx]["i"], data[idx]["a"][a_idx]["n"], data[idx]["a"][a_idx]["o"]),
            #         0, 0)
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

import numpy as np
from scipy import sparse
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
                                instances["%s_input_attr_%s_%s" % (legs[i], k1, k2-1)].append(n[k1][k2])
                                instances["%s_output_vertex_%s_%s" % (legs[i], k1, k2-1)].append(1)
        for k in instances:
            instances[k] = sparse.vstack(instances[k]).toarray()
        yield instances


def get_context_graph(idx):
    graph = {}
    graph["vertex_pub"] = [idx]
    graph["vertex_author"] = [a["i"] for a in data[idx]["a"]]
    graph["input_attr_name"] = []
    graph["input_attr_org"] = []
    graph["input_attr_title"] = []
    graph["input_attr_venue"] = []
    return graph


def run():
    num_node = len(data)
    node_attr_mapping = {
        "pub": ["t", "v", "k"],
        "author": ["n", "o"]
    }
    path_type = ["author", "pub", "author"]
    attr_shape = {}
    for t in ["n", "o", "t", "v", "k"]:
        attr_shape[t] = fvectors[t].shape[1]
    model = build_network(num_node, path_type, node_attr_mapping, attr_shape)
    epoch = 0
    for instances in gen_batch():
        print(len(instances["output_alignment"]))
        if len(instances["output_alignment"]) == 0:
            continue
        print(epoch)
        epoch += 1
        loss = model.fit(instances)
        print(loss)