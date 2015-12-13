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
    return " ".join(string.lower().replace(".", "").replace("-", "").replace("|", " ").strip().split())


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
                    for w in word_tokenize(item["venue"]["raw"]):
                        venue_df[stemmer.stem(w)] += 1
                if "authors" in item:
                    for a in item:
                        if "org" in a and len(a["org"]):
                            for w in word_tokenize(a["org"]):
                                aff_df[stemmer.stem(w)] += 1
