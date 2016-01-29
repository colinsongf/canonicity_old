import pickle
from collections import defaultdict as dd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from scipy import sparse
import numpy as np

stemmer = PorterStemmer()

def clean_name(name):
    x = [k.replace(".", "").replace("-", "").strip() for k in name.lower().split(" ")]
    return " ".join(x).strip()

def get_eval_data():
    id_to_idx = pickle.load(open("data/id_to_idx.pkl", "rb"))
    label_data = pickle.load(open("data/person_pub_data.pkl", "rb"))
    eval_pairs = []
    eval_pairs_blocked = {}
    eval_ins_blocked = {}
    for d in label_data:
        label_to_id = dd(list)
        id_to_label = {}
        eval_pairs_blocked[d["name"]] = []
        eval_ins_blocked[d["name"]] = []
        for p in d["pubs"]:
            x = p['id'] + "." + str(p['offset'])
            if x in id_to_idx:
                label_to_id[p['label']].append(id_to_idx[x])
                id_to_label[id_to_idx[x]] = p["label"]
        nodes = list(id_to_label.keys())
        eval_ins_blocked[d["name"]] = nodes
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if id_to_label[nodes[i]] == id_to_label[nodes[i+1]]:
                    eval_pairs.append((nodes[i], nodes[i+1], 1))
                    eval_pairs_blocked[d["name"]].append((nodes[i], nodes[i+1], 1))
                else:
                    eval_pairs.append((nodes[i], nodes[i+1], 0))
                    eval_pairs_blocked[d["name"]].append((nodes[i], nodes[i+1], 0))

    with open("eval_pairs.pkl", "wb") as f_out:
        pickle.dump(eval_pairs, f_out)

    with open("eval_pairs_blocked.pkl", "wb") as f_out:
        pickle.dump(eval_pairs_blocked, f_out)

    with open("eval_ins_blocked.pkl", "wb") as f_out:
        pickle.dump(eval_ins_blocked, f_out)


def gen_feature():
    vocab = pickle.load(open("data/vocab.pkl", "rb"))
    token_to_idx = pickle.load(open("data/token_to_idx.pkl", "rb"))
    idx_to_token = pickle.load(open("data/idx_to_token.pkl", "rb"))
    fvectors = pickle.load(open("data/fvectors.pkl", "rb"))
    data = pickle.load(open("data/dblp_data_new.pkl", "rb"))

    cell = {}
    row = {}
    col = {}
    for t in ["k", "v", "t", "o", "n", "p_n", "a_n"]:
        row[t] = []
        col[t] = []
        cell[t] = []

    feature_length = {
        "k": 5000,
        "v": 100,
        "t": 900,
        "o": 600,
        "n": 5000,
        "p_n": 5000,
        "a_n": 5000
    }

    r = 0
    for item in data.values():
        if "a" in item:
            # title
            for w in word_tokenize(item["t"].lower()):
                sw = stemmer.stem(w)
                if sw in token_to_idx["t"]:
                    col["t"].append(token_to_idx["t"][sw] % feature_length["t"])
                    row["t"].append(r)
                    cell["t"].append(.1 / vocab["t"][sw])
            # keywords
            for w in item["k"]:
                sw = stemmer.stem(w.lower())
                if sw in token_to_idx["k"]:
                    col["k"].append(token_to_idx["k"][sw] % feature_length["k"])
                    row["k"].append(r)
                    cell["k"].append(.1 / vocab["k"][sw])
            # venue
            if item["v"].lower() in token_to_idx["v"]:
                col["v"].append(token_to_idx["v"][item["v"].lower()] % feature_length["v"])
                row["v"].append(r)
                cell["v"].append(.1 / vocab["v"][item["v"].lower()])
            # author names
            for a in item["a"]:
                if a["n"] in token_to_idx["n"]:
                    col["p_n"].append(token_to_idx["n"][a["n"].lower()] % feature_length["n"])
                    row["p_n"].append(r)
                    cell["p_n"].append(.1 / vocab["n"][a["n"].lower()])
        else:
            # name
            if item["n"] in token_to_idx["n"]:
                col["n"].append(token_to_idx["n"][item["n"].lower()] % feature_length["n"])
                row["n"].append(r)
                cell["n"].append(.1 / vocab["n"][item["n"].lower()])
            # org
            for w in word_tokenize(item["o"].lower()):
                sw = stemmer.stem(w)
                if sw in token_to_idx["o"]:
                    col["o"].append(token_to_idx["o"][sw] % feature_length["o"])
                    row["o"].append(r)
                    cell["o"].append(.1 / vocab["o"][sw])
            # coauthor
            for a in data[item["p"]]["a"]:
                if a["i"] != item["i"]:
                    if a["n"] in token_to_idx["n"]:
                        col["p_n"].append(token_to_idx["n"][a["n"].lower()] % feature_length["n"])
                        row["p_n"].append(r)
                        cell["p_n"].append(.1 / vocab["n"][a["n"].lower()])

        r += 1
        if r % 10000 == 0:
            print(r)

    for t in ["k", "v", "t", "o", "n", "p_n", "a_n"]:
        fvectors[t] = sparse.coo_matrix((cell[t], (row[t], col[t])),
                                        shape=(len(data), feature_length[t]),
                                        dtype=np.float32).tocsr()

    features = {
        "p": sparse.hstack([fvectors[k] for k in ["t", "k", "v", "p_n"]]).tocsr(),
        "a": sparse.hstack([fvectors[k] for k in ["o", "a_n"]]).tocsr()
    }

    with open("features_hashed.pkl", "wb") as f_out:
        pickle.dump(features, f_out)


def gen_anchor():
    data = pickle.load(open("dblp_data.pkl", "rb"))
    sorted_names = pickle.load(open("sorted_names.pkl", "rb"))
    name_to_idx = pickle.load(open("name_to_idx.pkl", "rb"))
    anchors = []
    for i, n in enumerate(sorted_names[:800000]):
        if i % 100 == 0:
            print(i, n, len(anchors))
        for i in range(len(name_to_idx[n])):
            ni = name_to_idx[n][i]
            for j in range(i+1, len(name_to_idx[n])):
                nj = name_to_idx[n][j]
                anchors.append((data[ni[0]]["a"][ni[1]]["i"], data[nj[0]]["a"][nj[1]]["i"]))

    with open("anchors.pkl", "wb") as f_out:
        pickle.dump(anchors, f_out)
