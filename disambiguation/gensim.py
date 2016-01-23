from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import pickle
from collections import defaultdict as dd
import numpy as np
import logging
logging.getLogger().setLevel(logging.DEBUG)

authors_map = pickle.load(open("authors_map.pkl", "rb"))
pub_author_map = pickle.load(open("pub_author_map.pkl", "rb"))
attr = pickle.load(open("attr.pkl", "rb"))

data = pickle.load(open("dblp_data.pkl", "rb"))
name_to_idx = pickle.load(open("name_to_idx.pkl", "rb"))
sorted_names = pickle.load(open("sorted_names.pkl", "rb"))
eval_pairs_blocked = pickle.load(open("eval_pairs_blocked.pkl", "rb"))
path_size = 10




graph = dd(list)
cnt = 0
for d in data:
    if "a" in data[d]:
        if cnt % 10000 == 0:
            print(cnt)
        cnt += 1
        for a in data[d]["a"]:
            graph[a["i"]].append(data[d]["i"])
            graph[data[d]["i"]].append(a["i"])
cnt = 0
for n in range(800000):
    name = sorted_names[n]
    if cnt % 10000 == 0:
        print(cnt, name)
    cnt += 1
    pubs = name_to_idx[name]
    for i in range(len(pubs)):
        p = pubs[i]
        p_i = data[p[0]]["a"][p[1]]["i"]
        for j in range(i+1, len(pubs)):
            p = pubs[j]
            p_j = data[p[0]]["a"][p[1]]["i"]
            graph[p_i].append(p_j)
            graph[p_j].append(p_i)



# for e in authors_map:
#     graph[e[0]].append(e[1])
#     graph[e[1]].append(e[0])
# for e in pub_author_map:
#     graph[e[0]].append(e[1])
#     graph[e[1]].append(e[0])


def get_label(n):
    d = data[n]
    if "a" in d:
        return "PUB_%s" % n
    else:
        return "AUTHOR_%s" % n


def to_labels(p):
    path = []
    for n in p:
        path.append(get_label(n))
    return path

def gen_graph_path():
    pathes = []
    cnt = 0
    nodes = list(graph.keys())
    for _ in range(1):
        np.random.shuffle(nodes)
        for n in nodes:
            # if cnt % 1000 == 0:
            #     print(cnt, n, len(pathes))
            if len(graph[n]) == 0:
                continue
            path = [n]
            for _ in range(path_size):
                path.append(np.random.choice(graph[path[-1]]))
            pathes.append(to_labels(path))
            cnt += 1
    return pathes
    #     i = 0
    #     while i < idx.shape[0]:
    #         n = nodes[idx[i]]
    #         if cnt % 1000 == 0:
    #             print(cnt, len(pathes))
    #         cnt += 1
    #         if len(graph[n]) == 0:
    #             continue
    #         path = [n]
    #         for _ in range(path_size):
    #             path.append(np.random.choice(graph[n]))
    #
    #         # pa = []
    #         # for p in path:
    #         #     s = str(p) + "_"
    #         #     s += data[p]["n"] + "_"
    #         #     for a in attr[p][1:]:
    #         #         if a:
    #         #             s += (" ".join(a) + "_")
    #         #     pa.append(s)
    #         pathes.append(path)
    #         i += 1
    # return pathes


pathes = gen_graph_path()
model = Word2Vec(pathes, size=100, window=5, min_count=5, workers=4)

def eval(model):
    for n in eval_pairs_blocked:
        if len(eval_pairs_blocked[n]) < 50:
            continue
        print(n)
        tp = .00001
        fp = .00001
        fn = .00001
        for p in eval_pairs_blocked[n]:
            if "AUTHOR_%s" % p[0] in model and "AUTHOR_%s" % p[1] in model:
                s = model.similarity("AUTHOR_%s" % p[0], "AUTHOR_%s" % p[1])
                if p[2] > 0:
                    if s > .3:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if s > .3:
                        fp += 1
        print("pre", tp / (tp + fp), "rec", tp / (tp + fn))

for i in range(50):
    print(i)
    pathes = gen_graph_path()
    model.train(pathes)
    eval(model)