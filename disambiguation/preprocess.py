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
    with codecs.open("../data/person_pub_data.json", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=4)


def gen_data():
    import json
    import codecs
    data = json.load(codecs.open("../data/person_pub_data.json", encoding="utf-8"))
