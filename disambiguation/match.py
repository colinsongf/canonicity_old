from pymongo import MongoClient
from collections import defaultdict as dd

db = MongoClient(host='aminer.org', port=30019).bigsci
db.authenticate('kegger_bigsci', 'datiantian123!@#')
people_col = db["people"]
nsfc_col = db["nsfc_experts"]
profiles = dd(lambda: dd(int))

cnt = 0
for item in nsfc_col.find():
    print(cnt, item["name"], item["expert_id"])
    for p in item["pubs"]:
        profile = people_col.find_one({"pubs": {"$elemMatch": {"i": p["i"], "r": p["r"]}}}, {"_id": 1, "name": 1})
        if not profile:
            continue
        profiles[item["expert_id"]][str(profile["_id"])] += 1
        if item["name"] != profile.get("name", "NO NAME"):
            print(profile.get("name", "NO NAME"))
    print(len(profiles[item["expert_id"]]))



