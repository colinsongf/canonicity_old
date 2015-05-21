from feature import FeatureVectorAffiliation
import logging
import CRFPP
import nltk

def get_tagger():
	tagger = CRFPP.Tagger("-m ../models/model.crf -v 3 -n2")
	tagger.clear()
	return tagger

class AffilaitonParser(object):
	def __init__(self):
		pass

	def process(self, data):
		blocks = []
		output = []
		data = data.strip()
		for line in data.split("\n"):
			tokens = nltk.word_tokenize(line)
			for tk in tokens:
				blocks.append(tk)# + " <affiliation>")
				output.append(tk)
			blocks.append("@newline")
			output.append(" ")
		print blocks
		
		features = FeatureVectorAffiliation.proc_lines(blocks, [])
		print features
		
		tagger = get_tagger()
		for f in features:
			tagger.add(f)
		tagger.parse()

		res = []
		for i in range(tagger.size() - 1):
			for j in range(tagger.xsize() -1):
				logging.info(tagger.x(i, j))
				res.append(tagger.x(i, j))
			res.append(tagger.y2(i))
		print res

if __name__ == "__main__":
	p = AffilaitonParser()
	p.process("School of Electrical and Electronic Engineering, Nanyang Technological University, Nanyang Avenue, Singapore 639798, Singapore")

