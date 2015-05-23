from feature import FeatureVectorAffiliation
import logging
import CRFPP
import nltk
from lexicon import lex

def get_tagger():
	tagger = CRFPP.Tagger("-m ../models/model.crf -v 3 -n2")
	tagger.clear()
	return tagger

def build_result(result):
	aff = {}
	cur_item = None
	cur_type = None
	for item in result:
		if item[1][0] == "I":
			if cur_item is not None:
				if not cur_type == "other":
					aff[cur_type] = cur_item
			cur_item = item[0]
			cur_type = item[1].split("<")[1].split(">")[0]
			if cur_type == "settlement":
				cur_type = "city"
		else:
			cur_item += (" " + item[0])
	if cur_item is not None:
		if not cur_type == "other":
			aff[cur_type] = cur_item
	return aff


class AffilaitonParser(object):
	def __init__(self):
		self.lexicon = lex

	def parse(self, features):
		print features
		tagger = get_tagger()
		for f in features:
			# print f
			tagger.add(str(f))
		tagger.parse()
		return tagger

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

		place_lexicon = self.lexicon.in_city_names(data)
		
		features = FeatureVectorAffiliation.proc_lines(blocks, [place_lexicon])
		tagger = self.parse(features)

		res = []
		for i in range(tagger.size()):
			# for j in range(tagger.xsize() -1):
			res.append((tagger.x(i, 0), tagger.y2(i)))
		return build_result(res)

if __name__ == "__main__":
	p = AffilaitonParser()
	p.process("Knowledge Engineering Group, Department of Computer Science and Technology, Tsinghua University, Beijing, China")
	p.process("Faculty of Information Technology, University of Technology, Sydney, Australia")
	p.process("Microsoft Search Labs, USA")
	p.process("University of Calabria, Italy")
	p.process("Knowledge Engineering Institute, University of Science and Technology Beijing, Beijing , China")
	p.process("School of Information and Computer Sciences, Center for Machine Learning and Intelligent Systems, University of California, Irvine")
	p.process("School of Computer and Information,Hefei University of Technology,Hefei ,China")