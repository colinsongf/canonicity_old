import argparse
import pickle
from .model import Canonicity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neg_rate', type=int, default=0)
    args = parser.parse_args()

    features = pickle.load(open("features.pkl", "rb"))
    anchors = pickle.load(open("anchors.pkl", "rb"))
    data = pickle.load(open("dblp_data_new.pkl", "rb"))
    schema = {
        "nodes": {}
    }
    for n in features:
        schema["nodes"][n] = features[n].shape[1]

    model = Canonicity(args, schema, data, features, anchors)