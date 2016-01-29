import pickle
from collections import defaultdict as dd
import numpy as np
from sklearn.cluster import AffinityPropagation, DBSCAN, AgglomerativeClustering

ins = pickle.load(open("data/eval_ins_blocked.pkl", "rb"))
label = pickle.load(open("data/eval_pairs_blocked.pkl", "rb"))
data = pickle.load(open("data/dblp_data_new.pkl", "rb"))
name_to_idx = pickle.load(open("data/name_to_idx.pkl", "rb"))
features = pickle.load(open("data/features_hashed.pkl", "rb"))
fvectors = pickle.load(open("data/fvectors.pkl", "rb"))
sorted_names = pickle.load(open("data/sorted_names.pkl", "rb"))

def clean_name(name):
    x = [k.replace(".", "").replace("-", "").strip() for k in name.lower().split(" ")]
    return " ".join(x).strip()

def eval(n, pred):
    tp = 0.0001
    fp = 0.0001
    fn = 0.0001
    for i, j, y in label[n]:
        if not (i in pred and j in pred):
            continue
        if pred[i] == pred[j]:
            if y == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y == 1:
                fn += 1
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * pre * rec / (pre + rec)
    return tp, fp, fn, pre, rec, f1

def eval_test(py, y):
    tp = 0.0001
    fp = 0.0001
    fn = 0.0001
    for i, x in enumerate(py):
        if x == 1:
            if x == y[i]:
                tp += 1
            else:
                fp += 1
        else:
            if y[i] == 1:
                fn += 1
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * pre * rec / (pre + rec)
    return tp, fp, fn, pre, rec, f1

def ghost():

    def gen_graph(n):
        n = clean_name(n)
        graph = dd(set)
        # rs = set(name_to_idx[n])
        names = set()
        s = dd(set)
        for x in name_to_idx[n]:
            p = data[x[0]]
            for i, c in enumerate(p["a"]):
                if x[1] != i and c["n"] != n:
                    names.add(c["n"])
                    graph[x].add(c["n"])
                    graph[c["n"]].add(x)
                    s[(x, c["n"])].add(x[0])
                    s[(c["n"], x)].add(x[0])
        pubs = dd(set)
        new_names = set()
        for name in names:
            new_names.add(name)
        while len(new_names) > 0:
            # print(len(new_names), len(names))
            tmp_names = set()
            for k in new_names:
                for a in name_to_idx[k]:
                    if k != n:
                        pubs[a[0]].add(k)
                        for i, c in enumerate(data[a[0]]["a"]):
                            if i == a[1]:
                                continue
                            name = c["n"]
                            if name not in names:
                                tmp_names.add(name)
                                names.add(name)
            new_names = tmp_names

        for p in pubs:
            pubs[p] = list(pubs[p])
            for i in range(len(pubs[p])):
                for j in range(i+1, len(pubs[p])):
                    n1, n2 = pubs[p][i], pubs[p][j]
                    graph[n1].add(n2)
                    graph[n2].add(n1)
                    s[(n1, n2)].add(p)
                    s[(n2, n1)].add(p)
        return graph, s

    result = {}
    for nk in ins:
        print(nk)
        n = clean_name(nk)
        if n in result:
            continue
        g, s = gen_graph(n)
        queue = []
        source = dd(set)
        nodes = set()
        sub_path = dd(list)
        path = dd(list)
        cnt = 0
        for r in name_to_idx[n]:
            # print(cnt, len(name_to_idx[n]), r)
            # queue.append(r)
            for v in g[r]:
                nodes.add(v)
                source[v].add(r)
                queue.append(v)
                sub_path[r].append((r, v))
            while len(queue) > 0:
                u = queue.pop()
                for v in g[u]:
                    if v in g[r]:
                        nodes.add(v)
                        source[v].add(r)
                        if s[(r, u)] == s[(u, v)] and len(s[(r, u)]) == 1:
                            sub_path[r].append((r, u, v))

        for v in nodes:
            ss = list(source[v])
            for i in range(len(ss)):
                for j in range(i+1, len(ss)):
                    pair = (ss[i], ss[i+1])
                    for pi in sub_path[pair[0]]:
                        if pi[-1] == v:
                            for pj in sub_path[pair[1]]:
                                if pj[-1] == v:
                                    new_p = list(pi) + list(reversed(pj[:-1]))
                                    path[(pi[0], pj[0])].append(new_p)

        cnt = 0
        id2idx = {}
        idx2id = {}
        for r in name_to_idx[n]:
            id2idx[r] = cnt
            idx2id[cnt] = r
            cnt += 1
        if len(idx2id) == 0:
            continue

        sim = np.zeros((cnt, cnt))
        for pair in path:
            ss = .0
            for p in path[pair]:
                ss += 1. / len(p)
            sim[id2idx[pair[0]], id2idx[pair[1]]] = ss
            sim[id2idx[pair[1]], id2idx[pair[0]]] = ss

        params = [-500, -300, -200, -100, -50, -25, -10, -5, -1, None]
        res = {}
        for p in params:
            af = AffinityPropagation(preference=p, affinity='precomputed').fit(sim)
            pred = {}
            for x, y in enumerate(af.labels_):
                z = idx2id[x]
                pred[data[z[0]]["a"][z[1]]["i"]] = y

            res[p] = eval(nk, pred)
            print(p, res[p])
        result[n] = res
    with open("ghost_result.pkl", "wb") as f_out:
        pickle.dump(result, f_out)

    test_names = pickle.load(open("test_names.pkl", "rb"))
    params_af = [-500, -300, -200, -100, -50, -25, -10, -5, -1, None]
    with open("ghost_result.txt", "w") as f_out:
        row = []
        for p in params_af:
            for k in ["pre", "rec", "f1"]:
                row.append("ghost_%s_%s" % (p, k))
        f_out.write(",".join(row) + "\n")
        for n in test_names:
            row = []
            n = clean_name(n[0])
            if n in result:
                for p in params_af:
                    row += result[n][p][3:]
            else:
                row = [0.0 for _ in range(30)]
            f_out.write(",".join(["{0:.2f}".format(r*100) for r in row]) + "\n")


import jellyfish
from scipy import spatial
def get_edit_dist(s1, s2):
    return jellyfish.levenshtein_distance(clean_name(s1), clean_name(s2))

def get_jaro_winkler(s1, s2):
    return jellyfish.jaro_winkler(clean_name(s2), clean_name(s2))

def get_jaccard(s1, s2):
    if type(s1) is not set:
        s1, s2 = set(clean_name(s1).split()), set(clean_name(s2).split())
    x = len(s1.intersection(s2))
    y = len(s1.union(s2))
    if y == 0:
        return 0
    return float(x) / y

def get_cosine(s1, s2):
    # s1, s2 = s1.toarray()[0], s2.toarray()[0]
    # c = spatial.distance.cosine(s1, s2)
    # if c == np.nan:
    #     return 0
    # else:
    #     return c
    return s1.dot(s2.transpose())[0, 0]

def extract_feature(ni, nj):
    # org edit dist
    # org jaccard
    # org jaro winkler
    # org tfidf cosine
    # title edit dist
    # title jaccard
    # title jaro winkler
    # title tfidf cosine
    # venue edit dist
    # venue jaccard
    # venue jaro winkler
    # venue tfidf cosine
    # keyword jaccard
    # keyword cosine
    # print(i, j, ni, nj)
    s = [0 for _ in range(14)]
    # org
    ii, ij = data[ni[0]]["a"][ni[1]]["i"], data[nj[0]]["a"][nj[1]]["i"]
    oi, oj = data[ni[0]]["a"][ni[1]]["o"], data[nj[0]]["a"][nj[1]]["o"]
    s[0] = get_edit_dist(oi, oj)
    s[1] = get_jaccard(oi, oj)
    s[2] = get_jaro_winkler(oi, oj)
    s[3] = get_cosine(fvectors["o"][ii], fvectors["o"][ij])
    # title
    ti, tj = data[ni[0]]["t"], data[nj[0]]["t"]
    s[4] = get_edit_dist(ti, tj)
    s[5] = get_jaccard(ti, tj)
    s[6] = get_jaro_winkler(ti, tj)
    s[7] = get_cosine(fvectors["t"][ni[0]], fvectors["t"][nj[0]])
    # venue
    vi, vj = data[ni[0]]["v"], data[nj[0]]["v"]
    s[8] = get_edit_dist(vi, vj)
    s[9] = get_jaccard(vi, vj)
    s[10] = get_jaro_winkler(vi, vj)
    s[11] = get_cosine(fvectors["v"][ni[0]], fvectors["v"][nj[0]])
    # keyword
    ki, kj = set(data[ni[0]]["k"]), set(data[nj[0]]["k"])
    s[12] = get_jaccard(ki, kj)
    s[13] = get_cosine(fvectors["k"][ni[0]], fvectors["k"][ni[1]])
    return s

import random
from sklearn.linear_model import LogisticRegression
def train():
    pos_inst = []
    neg_inst = []
    for i in range(100000):
        if i % 100 == 0:
            print(i)
        n = random.randint(0, 800000)
        n = sorted_names[n]
        if len(name_to_idx[n]) < 3:
            continue
        pair = random.sample(name_to_idx[n], 2)
        pos_inst.append(pair)
        for _ in range(10):
            m = random.randint(0, len(data)-1)
            d = data[m]
            if "a" in d:
                if len(d["a"]) == 0:
                    continue
                j = random.randint(0, len(d["a"])-1)
                neg_inst.append((pair[0], (m, j)))
            else:
                neg_inst.append((pair[1], (d["p"], d["f"])))
    x = []
    y = []
    for n in pos_inst:
        x.append(extract_feature(n[0], n[1]))
        y.append(1)
    for n in neg_inst:
        x.append(extract_feature(n[0], n[1]))
        y.append(0)
    lr = LogisticRegression().fit(x, y)
    return lr


def clustering():
    model = train()
    result = {}
    result_af = {}
    result_hac = {}
    for nk in ins:
        n = clean_name(nk)
        print(n)
        if n in result and n in result_af and n in result_hac:
            continue
        num_item = len(name_to_idx[n])
        if num_item < 2:
            continue
        idx2id = {}
        id2idx = {}
        sim = np.zeros((num_item, num_item))
        distance = np.zeros((num_item, num_item))
        y = []
        for i in range(num_item):
            idx2id[i] = name_to_idx[n][i]
            id2idx[name_to_idx[n][i]] = i
            for j in range(i+1, num_item):
                ni, nj = name_to_idx[n][i], name_to_idx[n][j]
                x = extract_feature(ni, nj)
                s = model.predict_proba([x])[0]
                distance[i, j] = s[0]
                distance[j, i] = s[0]
                sim[i, j] = s[1]
                sim[j, i] = s[1]
        dbscan = DBSCAN(metric='precomputed', min_samples=1).fit(distance)
        pred = {}
        for i, c in enumerate(dbscan.labels_):
            p, f = idx2id[i]
            pred[data[p]["a"][f]["i"]] = c
        res = eval(nk, pred)
        result[n] = res
        print("DBSCAN", res)

        params = [-500, -300, -200, -100, -50, -25, -10, -5, -1, None]
        res = {}
        for p in params:
            af = AffinityPropagation(preference=p, affinity='precomputed').fit(sim)
            pred = {}
            for x, y in enumerate(af.labels_):
                z = idx2id[x]
                pred[data[z[0]]["a"][z[1]]["i"]] = y

            res[p] = eval(nk, pred)
            print("Affinity", p, res[p])
            result_af[n] = res

        params = [1, 2, 4, 8, 16]
        res = {}
        for p in params:
            try:
                hac = AgglomerativeClustering(n_clusters=p, affinity="precomputed", linkage="average").fit(sim)
                pred = {}
                for x, y in enumerate(hac.labels_):
                    z = idx2id[x]
                    pred[data[z[0]]["a"][z[1]]["i"]] = y

                res[p] = eval(nk, pred)
                print("HAC", p, res[p])
                result_hac[n] = res
            except:
                pass
    cnt = []
    params_af = [-500, -300, -200, -100, -50, -25, -10, -5, -1, None]
    params_hac = [1, 2, 4, 8, 16]
    with open("cluster_result.txt", "w") as f_out:
        row = ["dbscan_pre", "dbscan_rec", "dbscan_f1"]
        for p in params_af:
            for k in ["pre", "rec", "f1"]:
                row.append("af_%s_%s" % (p, k))
        for p in params_hac:
            for k in ["pre", "rec", "f1"]:
                row.append("hac_%s_%s" % (p, k))
        f_out.write(",".join(row) + "\n")
        for n in cnt:
            n = clean_name(n[0])
            if n in result:
                row = list(result[n][3:])
                for p in params_af:
                    row += result_af[n][p][3:]
                for p in params_hac:
                    if n in result_hac and p in result_hac[n]:
                        row += result_hac[n][p][3:]
                    else:
                        row += [0, 0, 0]
            else:
                row = [0.0 for _ in range(48)]
            f_out.write(",".join(["{0:.2f}".format(r*100) for r in row]) + "\n")


def get_gt():
    cnt = []
    for n in label:
        pos = 0
        neg = 0
        for k in label[n]:
            if k[2] == 0:
                neg += 1
            else:
                pos += 1
        cnt.append((n, pos+neg, pos, neg))
    cnt = sorted(cnt, key=lambda x: x[3] * x[2], reverse=True)
    with open("gt.txt", "w") as f_out:
        for row in cnt:
            f_out.write("%s,%s,%s,%s\n" % row)
    f_out.close()
    test_names = cnt
    with open("test_names.pkl", "wb") as f_out:
        pickle.dump(test_names, f_out)
