import nltk
from bs4 import UnicodeDammit
class Matcher():
	def __init__(self):
		self.terms = {}


	def load_terms(self, path):
		num_terms = 0
		f_in = open(path)
		for line in f_in:
			line =line.strip().lower()
			num_terms += self.load_term(line)
		return num_terms


	def load_term(self, term):
		term = UnicodeDammit(term).markup
		tokens = nltk.word_tokenize(term)
		cur_terms = self.terms
		for tk in tokens:
			if len(tk) == 0:
				continue
			if not tk in cur_terms:
				cur_terms[tk] = {}
			cur_terms = cur_terms[tk]
		if not "#" in cur_terms:
			cur_terms["#"] = {}
		if cur_terms != self.terms:
			return 1
		return 0


	def match(self, text):
		results = []
		start_pos = []
		last_non_separator_pos = []
		current_pos = 0

		tokens = nltk.word_tokenize(text)
		t = []
		for tk in tokens:
			if tk.strip() == "":
				continue
			if tk[0] == "<" and tk[-1] == ">":
				current_pos += 1
				continue
			tk = tk.lower()

			# we try to complete opened matching
			i = 0
			new_t = []
			new_start_pos = []
			new_last_non_separator_pos = []
			for tt in t:
				if tk in tt:
					new_t.append(tt[tk])
					new_start_pos.append(start_pos[i])
					new_last_non_separator_pos(current_pos)
				if "#" in tt:
					start = start_pos[i]
					end = last_non_separator_pos[i]
					results.append((start, end))
				i += 1

			# we start new matching starting at the current token
			if tk in self.terms:
				new_t.append(self.terms[tk])
				new_start_pos.append(current_pos)
				new_last_non_separator_pos.append(current_pos)

			t = new_t
			start_pos = new_start_pos
			last_non_separator_pos = new_last_non_separator_pos
			current_pos += 1

		# test if the end of the string correspond to the end of a term
		i = 0
		for tt in t:
			if "#" in tt:
				start = start_pos[i]
				end = last_non_separator_pos[i]
				results.append((start, end))
			i += 1

		return results

	def match_batch(self, tokens):
		text = ""
		for tk in tokens:
			if tk.strip() != "@newline":
				text += (" " + tk)
				# if not " " in tk and not "\t" in tk:
				# 	text += " " + tk
				# else:
				# 	text += " "
		return self.match(text)
