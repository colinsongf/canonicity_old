import pickle
import random
from collections import defaultdict as dd

from keras.layers.core import Dense, Layer, Lambda,Activation
from keras.layers.core import initializations, activations
from keras.layers.embeddings import Embedding
from keras.models import Graph, Sequential
from theano import tensor as T

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


"""
num_node: number of entity node to be embedded
node_attr_mapping: attributes of each node type
attr_shape: original dimension of each attribute
embedding_dim: dimension of the embedding space
"""

def build_model(num_node, node_attr_mapping, attr_shape, embedding_dim=100):
    shared_embedding = Sequential()
    shared_embedding.add(Embedding(input_dim=num_node, output_dim=embedding_dim, input_length=2))

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

    shared_layer = Sequential()
    shared_layer.add(Embedding(input_dim=num_node, output_dim=embedding_dim, input_length=2))
    for n in node_attr_mapping:
        n0 = Sequential()
        n0.add()



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

    def dummy(X):
        return X
    Dummy = Lambda(dummy)

    model = Graph()
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
        for leg in ["left", "right"]:
            model.add_input(name=leg + '_vertex_' + str(i), input_shape=(1,))
        attr_layers = []
        for j, a in enumerate(node_attr_mapping[n]):
            for leg in ["left", "right"]:
                model.add_input(name="%s_input_attr_%s_%s" % (leg, i, j), input_shape=attr_shape[a])
            model.add_shared_node(layer=Dense(output_dim=embedding_dim),
                                  name="encoder_attr_%s_%s" % (i, j),
                                  inputs=["%s_input_attr_%s_%s" % (leg, i, j) for leg in ["left", "right"]],
                                  outputs=["%s_encoded_attr_%s_%s" % (leg, i, j) for leg in ["left", "right"]])
            attr_layers.append("_encoded_attr_%s_%s" % (i, j))

        # add vertex attributes loss
        for leg in ["left", "right"]:
            model.add_node(Dummy, name="embedding_attr_%s" % i,
                           inputs=[leg+layer_name for layer_name in attr_layers], merge_mode="sum")
            model.add_output("%s_output_vertex_%s" % (leg, i),
                             inputs=["%s_embedding_attr_%s" % (leg, i), "%s_embeding_vertex_%s" % (leg, i)],
                             merge_mode="dot")
            output_loss.append(("%s_output_vertex_%s" % (leg, i), "binary_crossentropy"))

    # add local context loss
    for leg in ["left", "right"]:
        for i in range(1, len(path_type)):
            model.add_output("%s_output_edge_%s_%s" % (leg, i-1, i),
                             inputs=["%s_embeding_vertex_%s" % (leg, i-1), "%s_embeding_vertex_%s" % (leg, i)],
                             merge_mode="dot")
            output_loss.append(("%s_output_edge_%s_%s" % (leg, i-1, i), "binary_crossentropy"))

    # add alignment loss
    model.add_output("output_alignment",
                     inputs=["left_embeding_vertex_0", "right_embeding_vertex_0"],
                     merge_mode="dot")
    output_loss.append(("output_alignment", "binary_crossentropy"))

    model.compile('sgd', loss=dict(output_loss))
    return model


def build_network(num_node, node_attr_mapping, attr_shape, embedding_dim=100):
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
    model.add_input(name='edge_left', input_shape=(1,))
    model.add_input(name='edge_right', input_shape=(1,))
    for n in node_type_idx:
        model.add_input(name='vertex_' + n, input_shape=(1,))
    # model.add_node("NODE_vertex", embedding(input_dim=num_node, output_dim=embedding_dim, input_length=2))
    model.add_shared_node(Embedding(input_dim=num_node, output_dim=embedding_dim, input_length=2),
                          "embedding",
                          inputs=['edge_left', 'edge_right'] + ["vertex_" + k for k in node_type_idx.keys()],
                          outputs=['edge_left_embedding', 'edge_right_embedding'] + ["vertex_embedding_" + k for k in node_type_idx.keys()])
    for a in attr_type_idx:
        model.add_input("input_attr_"+a, input_shape=attr_shape[a])
        model.add_node("NODE_attr_"+a, Dense(output_dim=embedding_dim), input="input_attr_"+a)
    for n in node_attr_mapping:
        for a in node_attr_mapping:
            model.add_node()
            model.add_output(name="out_attr_" + a + "_" + n,
                 inputs=["vertex_embedding_" + n, "NODE_attr_" + a], merge_mode="dot")
            output_loss.append(("out_attr_" + a + "_" + n, "binary_crossentropy"))

    model.add_output(name="out_edge", inputs=["edge_left_embedding", "edge_right_embedding"], merge_mode="dot")
    output_loss.append(("out_edge", "binary_crossentropy"))

    model.compile('sgd', loss=dict(output_loss))
    return model


def gen_pair():
    num_names = 800000
    num_node = len(data)
    neg_sample = 10
    while True:
        current_name = random.randint(num_names)
        name = sorted_names[current_name]
        pubs = name_to_idx[name]
        if len(pubs) < 2:
            continue
        true_pair = tuple([data[k[0]]["a"][k[1]]["i"] for k in random.sample(name_to_idx[name], 2)])
        samples = [(true_pair, 1)]
        for i in [(0, 1), (1, 0)]:
            for j in range(neg_sample):
                idx = random.randint(num_node)
                false_pair = [-1, -1]
                false_pair[i[0]] = true_pair[i[0]]
                if "a" in data[idx]:
                    false_pair[i[1]] = random.sample(data[idx]["a"], 1)["i"]
                else:
                    false_pair[i[1]] = idx
                samples.append((tuple(false_pair), 0))
        yield samples


def train_model(model):
    pass


def get_context_graph(idx):
    graph = {}
    # for n in node_attr_mapping:
    #     graph["vertex_" + n] = []
    #     for a in node_attr_mapping[n]:
    #         graph["input_attr_" + a] = []
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
        "pub": ["title", "venue"],
        "author": ["name", "org"]
    }
    model = build_network(num_node, node_attr_mapping)
    for samples in gen_pair():
        for pair, label in samples:
            left = get_context_graph(pair[0])
            right = get_context_graph(pair[1])
