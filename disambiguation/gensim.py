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

path_size = 10

graph = dd(list)
for e in authors_map:
    graph[e[0]].append(e[1])
    graph[e[1]].append(e[0])
for e in pub_author_map:
    graph[e[0]].append(e[1])
    graph[e[1]].append(e[0])

def gen_graph_path():
    pathes = []
    cnt = 0
    for _ in range(1):
        idx = np.random.permutation(len(graph))
        i = 0
        while i < idx.shape[0]:
            if cnt % 1000 == 0:
                print(cnt, len(pathes))
            cnt += 1
            path = []
            if len(graph[idx[i]]) == 0:
                continue
            path = [idx[i]]
            for _ in range(path_size):
                path.append(np.random.choice(graph[path[-1]]))

            pa = []
            for p in path:
                s = str(p) + "_"
                s += attr[p][0] + "_"
                for a in attr[p][1:]:
                    if a:
                        s += (" ".join(a) + "_")
                pa.append(s)
            pathes.append(pa)
            i += 1
    return pathes


pathes = gen_graph_path()
model = Word2Vec(pathes, size=100, window=5, min_count=5, workers=4)