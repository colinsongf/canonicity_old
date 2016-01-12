import json
import codecs
from collections import defaultdict as dd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from pymongo import MongoClient
from bson import ObjectId
import pickle

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

def text_process(data):
