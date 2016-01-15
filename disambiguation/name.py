import json
import codecs
from collections import defaultdict as dd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from pymongo import MongoClient
from bson import ObjectId
import pickle
import numpy as np
from scipy import sparse

client = MongoClient('mongodb://yutao:911106zyt@yutao.us:30017/bigsci')
db = client["bigsci"]
col = db["publication_dupl"]


def exception_handler(iterator):
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            raise
        except Exception as e:
            print(e)
            pass

def get_name():
    segs = dd(int)
    names= dd(int)
    cnt = 0
    for item in exception_handler(col.find().skip(cnt)):
        if cnt % 1000 == 0:
            print(cnt)
        cnt += 1
        if "authors" in item:
            for a in item["authors"]:
                if not "name" in a or not a["name"]:
                    continue
                x = [k.replace(".", "").replace("-", "").strip() for k in a["name"].lower().split(" ")]
                for y in x:
                    segs[y] += 1
                names[" ".join(x)] += 1
    with open("names.pkl", "wb") as f_out:
        pickle.dump(names, f_out)
    with open("segs.pkl", "wb") as f_out:
        pickle.dump(segs, f_out)


def clean_name(name):
    x = [k.replace(".", "").replace("-", "").strip() for k in name.lower().split(" ")]
    return " ".join(x).strip()


def get_uniqueness(segs, name):
    score = 1
    if len(name.split()) > 2:
        return score
    for k in name.split():
        score *= segs[k]
    return score

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def load_data():
    num_node = 0
    index_to_id = {}
    data = {}
    cnt = 0
    name_to_idx = dd(list)
    id_to_idx = {}
    for item in exception_handler(col.find({"src": "dblp"}).skip(cnt)):
        if cnt % 1000 == 0:
            print(cnt)
        cnt += 1
        d = {
            "i": num_node,
            "id": str(item["_id"]),
            "t": clean_name(item.get("title", "") or ""),
            "v": clean_name(item.get("venue", {}).get("raw", "") or ""),
            "a": [],
            "k": item.get("keywords", []) or []
        }
        index_to_id[num_node] = str(item["_id"])
        id_to_idx[str(item["_id"])] = num_node
        cur_idx = num_node
        num_node += 1
        for i, a in enumerate(item.get("authors", [])):
            # if len(clean_name(a.get("name", "") or "")) < 2:
            #     continue
            author = {
                "i": num_node,
                "n": clean_name(a.get("name", "") or ""),
                "o": clean_name(a.get("org", "") or ""),
                "f": i,
                "p": cur_idx
            }
            d["a"].append(author)
            index_to_id[num_node] = (str(item["_id"]) + "." + str(i))
            name_to_idx[author["n"]].append((d["i"], i))
            id_to_idx[(str(item["_id"]) + "." + str(i))] = num_node
            data[num_node] = author
            num_node += 1
        data[cur_idx] = d

    segs = None
    with open("segs.pkl", "rb") as f_in:
        segs = pickle.load(f_in)

    sorted_names = sorted(name_to_idx.keys(), key=lambda x: get_uniqueness(segs, x))
    scores = [get_uniqueness(segs, x) for x in sorted_names]
    with open("scores.pkl", "wb") as f_out:
        pickle.dump(scores, f_out)

    with open("sorted_names.pkl", "wb") as f_out:
        pickle.dump(sorted_names, f_out)

    with open("dblp_data.pkl", "wb") as f_out:
        pickle.dump(data, f_out)

    with open("name_to_idx.pkl", "wb") as f_out:
        pickle.dump(name_to_idx, f_out)

    vocab = {}
    for t in ["n", "o", "t", "v", "k"]:
        vocab[t] = dd(int)

    cnt = 0
    for item in data.values():
        if cnt % 10000 == 0:
            print(cnt)
        cnt += 1
        if "a" in item:
            for w in word_tokenize(item["t"].lower()):
                sw = stemmer.stem(w)
                vocab["t"][sw] += 1
            for w in item["k"]:
                sw = stemmer.stem(w.lower())
                vocab["k"][sw] += 1
            vocab["v"][item["v"].lower()] += 1
        else:
            vocab["n"][item["n"]] += 1
            for w in word_tokenize(item["o"].lower()):
                sw = stemmer.stem(w)
                vocab["o"][sw] += 1


    token_to_idx = {}
    idx_to_token = {}
    for t in vocab:
        cnt = 0
        token_to_idx[t] = {}
        idx_to_token[t] = {}
        for token in vocab[t]:
            if vocab[t][token] > 1:
                token_to_idx[t][token] = cnt
                idx_to_token[t][cnt] = token
                cnt += 1

    with open("token_to_idx.pkl", "wb") as f_out:
        pickle.dump(token_to_idx, f_out)

    with open("idx_to_token.pkl", "wb") as f_out:
        pickle.dump(idx_to_token, f_out)

    with open("vocab.pkl", "wb") as f_out:
        pickle.dump(vocab, f_out)

    coordinate = {}
    cell = {}
    row = {}
    col = {}
    for t in ["n", "o", "t", "v", "k"]:
        row[t] = []
        col[t] = []
        for r, c in coordinate[t]:
            row[t].append(r)
            col[t].append(c)
        coordinate[t] = []
        cell[t] = []

    row = 0
    for item in data.values():
        if "a" in item:
            for w in word_tokenize(item["t"].lower()):
                sw = stemmer.stem(w)
                if sw in token_to_idx["t"]:
                    col = token_to_idx["t"][sw]
                    coordinate["t"].append((row, col))
                    cell["t"].append(.1 / vocab["t"][sw])
            for w in item["k"]:
                sw = stemmer.stem(w.lower())
                if sw in token_to_idx["k"]:
                    col = token_to_idx["k"][sw]
                    coordinate["k"].append((row, col))
                    cell["k"].append(.1 / vocab["k"][sw])
            if item["v"].lower() in token_to_idx["v"]:
                col = token_to_idx["v"][item["v"].lower()]
                coordinate["v"].append((row, col))
                cell["v"].append(.1 / vocab["v"][item["v"].lower()])
        else:
            if item["n"] in token_to_idx["n"]:
                col = token_to_idx["n"][item["n"].lower()]
                coordinate["n"].append((row, col))
                cell["n"].append(.1 / vocab["n"][item["n"].lower()])
            for w in word_tokenize(item["o"].lower()):
                sw = stemmer.stem(w)
                if sw in token_to_idx["o"]:
                    col = token_to_idx["o"][sw]
                    coordinate["o"].append((row, col))
                    cell["o"].append(.1 / vocab["o"][sw])
        row += 1
        if row % 10000 == 0:
            print(row)

    from scipy import sparse
    fvectors = {}
    for t in ["n", "o", "t", "v", "k"]:
        fvectors[t] = sparse.coo_matrix((cell[t], (row[t], col[t])),
                                        shape=(len(data), len(idx_to_token[t])),
                                        dtype=np.float32).tocsr()

    with open("fvectors.pkl", "wb") as f_out:
        pickle.dump(fvectors, f_out)



def plot_name(scores):
    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    mu, sigma = 100, 15
    x = scores

    # the histogram of the data
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    plt.axis([40, 160, 0, 0.03])
    plt.grid(True)

    plt.show()

