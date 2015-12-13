__author__ = 'yutao'


def clean_string(string):
    if not string:
        return ""
    return " ".join(string.lower().replace(".", "").replace("-", " ").replace("|", " ").strip().split())


def merge_institution():
    from pymongo import MongoClient
    from collections import defaultdict as dd

    client = MongoClient('mongodb://yutao:911106zyt@yutao.us:30017/bigsci')
    db = client["bigsci"]
    col = db["institution_all"]
    cnt = 0

    inst = dd(list)
    for item in col.find():
        inst[clean_string(item["name"])].append(str(item["_id"]))
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)


def dupe_aff(aff_data):
    import dedupe
    import logging

    log_level = logging.DEBUG
    logging.getLogger().setLevel(log_level)

    fields = [
        {'field': 'name', 'type': 'String'},
    ]
    data = {}
    cnt = 0
    for aff in aff_data:
        data[cnt] = {
            "id": cnt,
            "name": aff
        }
        cnt += 1
    deduper = dedupe.Dedupe(fields)
    deduper.sample(data, 15000)
    print('starting active labeling...')
    dedupe.consoleLabel(deduper)
    deduper.train()
    print('blocking...')
    threshold = deduper.threshold(data, recall_weight=2)

    # `match` will return sets of record IDs that dedupe
    # believes are all referring to the same entity.
    
    print('clustering...')
    clustered_dupes = deduper.match(data, threshold)
    
    print('# duplicate sets', len(clustered_dupes))

    # ## Writing Results
    
    # Write our original data back out to a CSV with a new column called 
    # 'Cluster ID' which indicates which records refer to each other.
    
    cluster_membership = {}
    cluster_id = 0
    for (cluster_id, cluster) in enumerate(clustered_dupes):
        id_set, scores = cluster
        cluster_d = [data[c] for c in id_set]
        canonical_rep = dedupe.canonicalize(cluster_d)
        for record_id, score in zip(id_set, scores) :
            cluster_membership[record_id] = {
                "cluster id" : cluster_id,
                "canonical representation" : canonical_rep,
                "confidence": score
            }
    
    singleton_id = cluster_id + 1