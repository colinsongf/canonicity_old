import pickle
from collections import defaultdict as dd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from scipy import sparse
import numpy as np

stemmer = PorterStemmer()

def get_eval_data():
    id_to_idx = pickle.load(open("id_to_idx.pkl", "rb"))
    label_data = pickle.load(open("person_pub_data.pkl", "rb"))
    eval_pairs = []
    eval_pairs_blocked = {}
    eval_ins_blocked = {}
    for d in label_data:
        label_to_id = dd(list)
        eval_pairs_blocked[d["name"]] = []
        eval_ins_blocked[d["name"]] = []
        for p in d["pubs"]:
            x = p['id'] + "." + str(p['offset'])
            if x in id_to_idx:
                label_to_id[p['label']].append(id_to_idx[x])
        for l in label_to_id:
            for i in range(len(label_to_id[l])):
                eval_ins_blocked[d["name"]].append(label_to_id[l][i])
                for j in range(i+1, len(label_to_id[l])):
                    eval_pairs.append((label_to_id[l][i], label_to_id[l][j], 1))
                    eval_pairs_blocked[d["name"]].append((label_to_id[l][i], label_to_id[l][j], 1))
        for l1 in range(len(label_to_id)):
            for l2 in range(l1+1, len(label_to_id)):
                for n1 in label_to_id[l1]:
                    for n2 in label_to_id[l2]:
                        eval_pairs.append((n1, n2, 0))
                        eval_pairs_blocked[d["name"]].append((n1, n2, 0))

    with open("eval_pairs.pkl", "wb") as f_out:
        pickle.dump(eval_pairs, f_out)

    with open("eval_pairs_blocked.pkl", "wb") as f_out:
        pickle.dump(eval_pairs_blocked, f_out)

    with open("eval_ins_blocked.pkl", "wb") as f_out:
        pickle.dump(eval_ins_blocked, f_out)


def gen_feature():
    vocab = pickle.load(open("vocab.pkl", "rb"))
    token_to_idx = pickle.load(open("token_to_idx.pkl", "rb"))
    idx_to_token = pickle.load(open("idx_to_token.pkl", "rb"))
    fvectors = pickle.load(open("fvectors.pkl", "rb"))
    data = pickle.load(open("dblp_data.pkl", "rb"))

    cell = {}
    row = {}
    col = {}
    for t in ["p_n", "a_n"]:
        row[t] = []
        col[t] = []
        cell[t] = []

    r = 0
    for item in data.values():
        if "a" in item:
            for a in item["a"]:
                if a["n"] in token_to_idx["n"]:
                    col["p_n"].append(token_to_idx["n"][a["n"].lower()])
                    row["p_n"].append(r)
                    cell["p_n"].append(.1 / vocab["n"][a["n"].lower()])
        else:
            for a in data[item["p"]]["a"]:
                if a["i"] != item["i"]:
                    if a["n"] in token_to_idx["n"]:
                        col["p_n"].append(token_to_idx["n"][a["n"].lower()])
                        row["p_n"].append(r)
                        cell["p_n"].append(.1 / vocab["n"][a["n"].lower()])

        r += 1
        if r % 10000 == 0:
            print(r)

    for t in ["p_n", "a_n"]:
        fvectors[t] = sparse.coo_matrix((cell[t], (row[t], col[t])),
                                        shape=(len(data), len(idx_to_token["n"])),
                                        dtype=np.float32).tocsr()

    features = {
        "a": sparse.hstack([fvectors[k] for k in ["t", "k", "v", "p_n"]]).tocsr(),
        "p": sparse.hstack([fvectors[k] for k in ["o", "a_n"]]).tocsr()
    }

    with open("features.pkl", "wb") as f_out:
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
