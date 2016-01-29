import pickle
from collections import defaultdict as dd
import gensim
import numpy as np
from gensim.models.word2vec import Word2Vec

features = pickle.load(open("data/features_hashed.pkl", "rb"))
anchors = pickle.load(open("data/anchors.pkl", "rb"))
data = pickle.load(open("data/dblp_data_new.pkl", "rb"))
test_data = pickle.load(open("data/eval_pairs.pkl", "rb"))
label = pickle.load(open("data/eval_pairs_blocked.pkl", "rb"))

def get_label(t, i):
    return "%s__%s" % (t, i)

def recover_label(l):
    x = l.split("__")
    if x[0] in {"p", "a"}:
        x[1] = int(x[1])
    return x[0], x[1]

def gen_graph():
    graph = dd(list)
    cnt = 0
    for i in data:
        if cnt % 1000 == 0:
            print(cnt)
        cnt += 1
        d = data[i]
        if "a" in d:
            ld = get_label("p", d["i"])
            for a in d["a"]:
                la = get_label("a", a["i"])
                graph[ld].append(la)
                graph[la].append(ld)
                if len(a["o"]) > 2:
                    lo = get_label("o", a["o"])
                    graph[lo].append(la)
                    graph[la].append(lo)
            for k in d.get("k", []):
                lk = get_label("k", k)
                graph[lk].append(ld)
                graph[ld].append(lk)
            if len(d["v"]) > 2:
                lv = get_label("v", d["v"])
                graph[lv].append(ld)
                graph[ld].append(lv)
    for a in anchors:
        la1 = get_label("a", data[a[0]]["i"])
        la2 = get_label("a", data[a[1]]["i"])
        graph[la1] = graph[la2]
        graph[la2] = graph[la1]
    with open("dw_graph.pkl", "wb") as f_out:
        pickle.dump(graph, f_out)

    return graph

def eval(model):
    for n in label:
        if len(label[n]) < 50:
            continue
        print(n)
        tp = .00001
        fp = .00001
        fn = .00001
        for p in label[n]:
            la1, la2 = get_label("a", p[0]), get_label("a", p[1])
            if la1 in model and la2 in model:
                s = model.similarity(la1, la2)
                if p[2] > 0:
                    if s > .3:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if s > .3:
                        fp += 1
        print("pre", tp / (tp + fp), "rec", tp / (tp + fn))

def deepwalk():
    print("load graph...")
    graph = pickle.load(open("data/vocab.pkl", "rb"))
    print("graph loaded")

    def gen_graph_path(graph, path_size=10):
        pathes = []
        cnt = 0
        nodes = list(graph.keys())
        for _ in range(1):
            np.random.shuffle(nodes)
            for n in nodes:
                if cnt % 1000 == 0:
                    print(cnt)
                if len(graph[n]) == 0:
                    continue
                path = [n]
                for _ in range(path_size):
                    path.append(np.random.choice(graph[path[-1]]))
                pathes.append(path)
                cnt += 1
        return pathes

    pathes = gen_graph_path(graph)
    model = Word2Vec(pathes, size=100, window=5, min_count=5, workers=4)

    for i in range(50):
        print(i)
        pathes = gen_graph_path(graph)
        model.train(pathes)
        eval(model)


if __name__ == '__main__':
    deepwalk()