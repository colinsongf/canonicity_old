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

def exception_handler(iterator):
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            raise
        except Exception as e:
            print(e)
            pass

def get_name():
    segs = dd(int)
    names= dd(int)
    cnt = 0
    for item in exception_handler(col.find().skip(cnt)):
        if cnt % 1000 == 0:
            print(cnt)
        cnt += 1
        if "authors" in item:
            for a in item["authors"]:
                if not "name" in a or not a["name"]:
                    continue
                x = [k.replace(".", "").replace("-", "").strip() for k in a["name"].lower().split(" ")]
                for y in x:
                    segs[y] += 1
                names[" ".join(x)] += 1