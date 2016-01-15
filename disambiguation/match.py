from pymongo import MongoClient
from collections import defaultdict as dd
from bson import ObjectId
import requests

db = MongoClient(host='aminer.org', port=30019).bigsci
db.authenticate('kegger_bigsci', 'datiantian123!@#')
people_col = db["people"]
nsfc_col = db["nsfc_experts"]
profiles = dd(lambda: dd(int))

db2 = MongoClient(host='yutao.yt', port=30017).bigsci
db2.authenticate('kegger_bigsci', 'datiantian123!@#')
people_col2 = db["people"]

item = people_col2.find_one({"_id": ObjectId("53f46a3edabfaee43ed05f08")})



cnt = 112630
for item in nsfc_col.find().skip(cnt):
    print(cnt, item["name"], item["expert_id"])
    for p in item.get("pubs", []):
        profile = people_col.find_one({"pubs": {"$elemMatch": {"i": p["i"], "r": p["r"]}}}, {"_id": 1, "name": 1})
        if not profile:
            continue
        profiles[item["expert_id"]][str(profile["_id"])] += 1
        if item["name"] != profile.get("name", "NO NAME"):
            print(profile.get("name", "NO NAME"))
    print(len(profiles[item["expert_id"]]))
    cnt += 1


import pickle
with open("nsfc_profiles.pkl", "wb") as f_out:
    pickle.dump(profiles, f_out)


def merge_person(mid, tids):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIyLUpDQ2g0blg1OW4zVFUrWUJoUVV3ZHNIOTI1Q05tQkR3VWN5RU10WmhaVFVaRjBHRnhqRFB4K3phT2tROEU1VGNPeWhvUHNiTyt1WGtqdGVWXC9xaE1tUmE5QkM4Zm1wZ0RJbE09IiwiaXNzIjoicGxheS1hbmd1bGFyLXNpbGhvdWV0dGUiLCJleHAiOjE0NTUyODY4MzIsImlhdCI6MTQ1MjY5NDgzMiwianRpIjoiYzM1ZjVjNWQ5ZDMwZWFmYmE3MzJhNDYzNTg3ZWNlNzg2NGM5YWE5YTA1NjA5MWRhYjgxOTU2MTIyZDE5M2I5MTc0NDNkMjcyNzBhN2Q0MGEyZDNhM2E4ODFiN2UwMmU3Y2IxNjMzMzliNjQwZTNmNDcyZTM1ZDU1ZDc2MjA2NjFhMzYyOTZiZGJkYjczYzRjMTcwOWZjY2I3M2M2Y2MwMzU3MGIzMzM1NDFhZDU3ZWVlNDI5OTQ4ZTM2YTU4OGUwN2YxNTUyODVhZmRmOTc2ZWJiNTMzZmY0YzljYjE3MjM3NjI2ZjkyY2VlNDI3OGU5MmIwOTFhZTljOTBkNDRhNCJ9.AYS2fMYlYGP8CyJAN0tRFfVXkDy1pIjsnp_amNYKk_g"
    }
    resp = requests.post("https://api.aminer.org/api/fusion/person/merge/%s" % mid, json={"r": [{"i": tid} for tid in tids]}, headers=headers)
    print(resp.text)

import pickle
profiles = pickle.load(open("nsfc_profiles.pkl", "rb"))
skip = 116683
cnt = 0
for i in profiles:
    print(cnt, i)
    cnt += 1
    if cnt < skip:
        continue
    sorted_pubs = sorted(profiles[i].items(), key=lambda x: x[1], reverse=True)
    if len(sorted_pubs) == 0:
        continue
    mid = None
    flag = False
    for p in sorted_pubs:
        item = people_col.find_one({"_id": ObjectId(p[0])})
        if item is not None:
           mid = p[0]
           if item.get("nsfc_id", 0) == i:
               flag = True
           break
    if flag:
        continue

    if mid is None:
        continue

    tids = []
    for p in sorted_pubs:
        if p[1] < 10 and p[0] != mid:
            tids.append(p[0])
    cur = 0
    while cur < len(tids):
        merge_person(mid, tids[cur: cur+80])
        cur += 80
    item = people_col.find_one({"_id": ObjectId(mid)})
    item["nsfc_id"] = i
    print(mid)
    people_col.save(item)

nsfc = dd(list)
cnt = 0
for item in people_col.find():
    if cnt % 10000 == 0:
        print(cnt)
    cnt += 1
    if "nsfc_id" in item:
        nsfc[item["nsfc_id"]].append((str(item["_id"]), len(item["pubs"])))

dup = []
for n in nsfc:
    if len(nsfc[n]) > 1:
        dup.append(n)
