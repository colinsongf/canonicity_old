__author__ = 'yutao'

def collect_data():
    import json
    import codecs
    import requests
    data = json.load(codecs.open("../data/person_rst.json", encoding="utf-8"))
    cnt = 0
    for p in data["person"]:
        print(cnt)
        cnt += 1
        print(p["FullName"])
        for pub in p["publication"]:
            try:
                resp = requests.get("https://api.aminer.org/api/search/pub?query=%s" % pub["title"]).json()
                if len(resp["result"]) == 0:
                    continue
                res = resp["result"][0]
                print(pub["title"])
                print(pub["authors"])
                if len(res['authors']) == 0:
                    continue
                if res["authors"][0]["name"][0] == pub["authors"].split(",")[0][0]:
                    pub["data"] = res
                    print(res["title"])
                    print(res["authors"][0]["name"])
            except:
                pass
    with codecs.open("../data/person_pub_data.json", "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=4)

def get_author_email():
    from pymongo import MongoClient
    from collections import defaultdict as dd

    client = MongoClient('mongodb://yutao:911106zyt@yutao.us:30017/bigsci')
    db = client["bigsci"]
    col = db["publication_dupl"]

    emails = dd(list)
    cnt = 0
    cnt1 = 0
    cnt2 = 0
    flag = True
    while flag:
        flag = False
        try:
            for item in col.find().skip(cnt2):
                cnt2 += 1
                if "authors" in item:
                    for i, a in enumerate(item["authors"]):
                        cnt1 += 1
                        if "email" in a and "@" in a["email"]:
                            emails[a["email"]].append((str(item["_id"]), i))
                            cnt += 1
                            print(len(emails), cnt, cnt1, float(cnt) / len(emails), float(cnt) / cnt1)
        except:
            cnt2 += 1
            flag = True



def clean_string(string):
    if not string:
        return ""
    s = " ".join(string.lower().replace(".", "").replace("-", "").replace("|", " ").strip().split())
    return " ".join(sorted(s.split(" ")))



def get_vocab():
    import json
    import codecs
    from collections import defaultdict as dd
    from nltk import word_tokenize
    from nltk.stem.porter import PorterStemmer
    from pymongo import MongoClient
    from bson import ObjectId

    client = MongoClient('mongodb://yutao:911106zyt@yutao.us:30017/bigsci')
    db = client["bigsci"]
    col = db["publication_dupl"]

    stemmer = PorterStemmer()
    data = json.load(codecs.open("../data/person_pub_data.json", encoding="utf-8"))["person"]
    title_df = dd(int)
    venue_df = dd(int)
    aff_df = dd(int)
    for d in data:
        for p in d["publication"]:
            if "data" in p:
                i = p["data"]["id"]
                item = col.find_one({"_id": ObjectId(i)})
                for w in word_tokenize(item["title"].lower()):
                    title_df[stemmer.stem(w)] += 1
                if "venue" in item and "raw" in item["venue"]:
                    for w in word_tokenize(item["venue"]["raw"].lower()):
                        venue_df[stemmer.stem(w)] += 1
                if "authors" in item:
                    for a in item["authors"]:
                        if "org" in a and a["org"] and len(a["org"]):
                            for w in word_tokenize(a["org"].lower()):
                                aff_df[stemmer.stem(w)] += 1


def clean_data():
    import json
    import codecs
    from bson import ObjectId
    from pymongo import MongoClient
    client = MongoClient('mongodb://yutao:911106zyt@yutao.us:30017/bigsci')
    db = client["bigsci"]
    col = db["publication_dupl"]
    new_data = []
    cnt = 0
    data = json.load(codecs.open("../data/person_pub_data.json", encoding="utf-8"))["person"]
    for d in data:
        person = {
            "name": d["FullName"],
            "pubs": []
        }
        print(person["name"], cnt)
        for pub in d["publication"]:
            if "data" in pub:
                offset = -1
                item = col.find_one({"_id": ObjectId(pub["data"]["id"])})
                authors = {}
                for i, a in enumerate(item["authors"]):
                    o = None
                    n = None
                    if "org" in a and a["org"] and len(a["org"]) > 1:
                        o = a["org"]
                    if "name" in a and a["name"] and len(a["name"]) > 1:
                        n = a["name"]
                        if clean_string(n) == clean_string(person["name"]):
                            offset = i
                    authors[i] = {
                        "name": n,
                        "aff": o
                    }
                if offset >= 0:
                    person["pubs"].append({
                        "year": pub["year"],
                        "label": int(pub["label"]),
                        "offset": offset,
                        "id": pub["data"]["id"],
                        "title": item["title"],
                        "authors": authors,
                        "venue": None if not ("venue" in item and "raw" in item["venue"]) else item["venue"]["raw"]
                    })
                    cnt += 1
        new_data.append(person)

    data = new_data

    cnt_p = 0
    for d in data:
        for pub in d["pubs"]:
            pub["idx"] = cnt_p
            cnt_p += 1

    cnt_a = cnt_p
    for d in data:
        for pub in d["pubs"]:
            for a in pub["authors"].values():
                a["idx"] = cnt_a
                cnt_a += 1

    import pickle
    with open("person_pub_data.pkl", "wb") as f_out:
        pickle.dump(data, f_out)

def get_data():
    from collections import defaultdict as dd
    from nltk import word_tokenize
    from nltk.stem.porter import PorterStemmer
    import pickle

    stemmer = PorterStemmer()
    data = pickle.load(open("../data/person_pub_data.pkl", "rb"))
    title_df = dd(int)
    venue_df = dd(int)
    aff_df = dd(int)

    for d in data:
        for item in d["pubs"]:
            for w in word_tokenize(item["title"].lower()):
                title_df[stemmer.stem(w)] += 1
            if item["venue"]:
                for w in word_tokenize(item["venue"].lower()):
                    venue_df[stemmer.stem(w)] += 1
            for a in item["authors"].values():
                if a["aff"]:
                    for w in word_tokenize(a["aff"].lower()):
                        aff_df[stemmer.stem(w)] += 1

    with open("aff_vocab.pkl", "wb") as f_out:
        pickle.dump(list(aff_df.items()), f_out)
    with open("venue_vocab.pkl", "wb") as f_out:
        pickle.dump(list(venue_df.items()), f_out)
    with open("title_vocab.pkl", "wb") as f_out:
        pickle.dump(list(title_df.items()), f_out)

    pub_author_map = []
    authors_map = []
    for d in data:
        labels = dd(list)
        for item in d["pubs"]:
            labels[item["label"]].append(item["authors"][item["offset"]]["idx"])
            for a in item["authors"].values():
                pub_author_map.append((item["idx"], a["idx"]))
        for l in labels:
            for i in range(len(labels[l])):
                for j in range(i+1, len(labels[l])):
                    authors_map.append((labels[l][i], labels[l][j]))
    with open("pub_author_map.pkl", "wb") as f_out:
        pickle.dump(pub_author_map, f_out)
    with open("authors_map.pkl", "wb") as f_out:
        pickle.dump(authors_map, f_out)

    attr = [None for i in range(25102)]
    for d in data:
        for pub in d["pubs"]:
            title, venue = pub["title"], pub["venue"]
            if title:
                title = [stemmer.stem(w) for w in word_tokenize(pub["title"].lower())]
            if venue:
                venue = [stemmer.stem(w) for w in word_tokenize(pub["venue"].lower())]
            attr[pub["idx"]] = ("pub", title, venue)
            for a in pub["authors"].values():
                name, aff = a["name"], a["aff"]
                if name:
                    name = [stemmer.stem(w) for w in word_tokenize(a["name"].lower())]
                if aff:
                    aff = [stemmer.stem(w) for w in word_tokenize(a["aff"].lower())]
                attr[a["idx"]] = ("author", name, aff)
    with open("attr.pkl", "wb") as f_out:
        pickle.dump(attr, f_out)
