from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Merge, LambdaMerge, Lambda, Siamese
from keras.models import Graph
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
import pickle
from theano import tensor as T
import random



def build_network(num_node, edge_types, node_attr_mapping, attr_shape, embedding_dim=100):
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
    model.add_input(name='EDGE_LEFT', input_shape=(1,))
    model.add_input(name='EDGE_RIGHT', input_shape=(1,))
    for n in node_type_idx:
        model.add_input(name='VERTEX_' + n, input_shape=(1,))
    # model.add_node("NODE_VERTEX", Embedding(input_dim=num_node, output_dim=embedding_dim, input_length=2))
    model.add_shared_node(Embedding(input_dim=num_node, output_dim=embedding_dim, input_length=2),
                          "EMBEDDING",
                          inputs=['EDGE_LEFT', 'EDGE_RIGHT'] + ["VERTEX_" + k for k in node_type_idx.keys()],
                          outputs=['EDGE_LEFT_EMBEDDING', 'EDGE_RIGHT_EMBEDDING'] + ["VERTEX_EMBEDDING_" + k for k in node_type_idx.keys()])
    for a in attr_type_idx:
        model.add_input("INPUT_ATTR_"+a, input_shape=attr_shape[a])
        model.add_node("NODE_ATTR_"+a, Dense(output_dim=embedding_dim), input="INPUT_ATTR_"+a)
    for n in node_attr_mapping:
        for a in node_attr_mapping:
            model.add_output(name="OUT_ATTR_" + a + "_" + n,
                 inputs=["VERTEX_EMBEDDING_" + n, "NODE_ATTR_" + a], merge_mode="dot")
            output_loss.append(("OUT_ATTR_" + a + "_" + n, "binary_crossentropy"))

    model.add_output(name="OUT_EDGE", inputs=["EDGE_LEFT_EMBEDDING", "EDGE_RIGHT_EMBEDDING"], merge_mode="dot")
    output_loss.append(("OUT_EDGE", "binary_crossentropy"))

    model.compile('sgd', loss=dict(output_loss))
