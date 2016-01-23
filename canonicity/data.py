import pickle
from collections import defaultdict as dd

def load_data():
    pass


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