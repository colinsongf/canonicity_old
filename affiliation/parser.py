from feature import FeatureVectorAffiliation
import logging
import CRFPP
import nltk

class AffilaitonParser(object):
	def __init__(self):
		pass

	@static_method
	def get_tagger():
		tagger = CRFPP.Tagger("-m ../models/model.crf -v 3 -n2")
		tagger.clear()
		return tagger

	def process(data):
		blocks = []
		output = []
		data = data.strip()
		for line in data.split("\n"):
			tokens = nltk.word_tokenize(line)
			for tk in tokens:
				blocks.append(tk + " <affiliation>")
				output.append(tk)
			blocks.append("@newline")
			output.append(" ")
		
		features = FeatureVectorAffiliation.proc_lines(blocks, [])
		
		tagger = get_tagger()
		tagger.add(features)
		tagger.parse()

		res = []
		for i in range(tagger.size() - 1):
			for j in range(tagger.xsize() -1):
				logging.info(tagger.x(i, j))
				res.append(tagger.x(i, j))
			res.append(tagger.y2(i))

