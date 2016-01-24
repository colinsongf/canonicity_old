import argparse
import pickle
from canonicity.model import Canonicity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neg_rate', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--learning_rate', type=int, default=0.1)
    args = parser.parse_args()

    features = pickle.load(open("data/features.pkl", "rb"))
    anchors = pickle.load(open("data/anchors.pkl", "rb"))
    data = pickle.load(open("data/dblp_data_new.pkl", "rb"))
    schema = {
        "nodes": {}
    }
    for n in features:
        schema["nodes"][n] = features[n].shape[1]

    model = Canonicity(args, schema, data, features, anchors)
    model.build()