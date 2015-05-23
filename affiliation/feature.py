import utils
from lexicon import lex

class FeatureVectorAffiliation(object):
	def __init__(self):
		self.string = None
		self.label = None
		self.line_stat = None # LINESTART, LINEIN, LINEEND
		self.bold = False
		self.italic = False
		self.capital = None # INITCAP, ALLCAPS, NOCAPS
		self.digit = None # ALLDIGIT, CONTAINDIGIT, NODIGIT
		self.single_char = False
		self.proper_name = False
		self.common_name = False
		self.first_name = False
		self.location_name = False
		self.country_name = False
		self.punct_type = None
		self.word_shape = None # NOPUNCT, OPENBACKET, ENDBACKET, DOT, COMMA, HYPHEN, QUOTE, PUNCT

	@staticmethod
	def proc_line(line, line_stat, is_place):
		features = FeatureVectorAffiliation()
		tokens = line.split()
		for i, token in enumerate(tokens):
			label = None
			if i+1 < len(tokens):
				label = tokens[i+1]
			features.string = token
			features.label = label
			features.line_stat = line_stat
			if len(token) == 1:
				features.single_char = True
			if utils.test_all_captial(token):
				features.capital = "ALLCAPS"
			elif utils.test_first_capital(token):
				features.capital = "INITCAP"
			else:
				features.capital = "NOCAPS"

			features.common_name = False
			features.proper_name = False
			features.location_name = is_place


			if utils.test_all_digit(token):
				features.digit = "ALLDIGIT"
			elif utils.test_contain_digit(token):
				features.digit = "CONTAINDIGIT"
			else:
				features.digit = "NODIGIT"

			features.punct_type = "NOPUNCT"
			if token == ",":
				features.punct_type = "COMMA"
			if token == "(" or token == "[":
				features.punct_type = "OPENBRACKET"
			if token == ".":
				features.punct_type = "DOT"
			if token == "-":
				features.punct_type = "HYPHEN"
			if token == "\"" or token == "\'" or token == "`":
				features.punct_type = "QUOTE"

			features.country_name = lex.is_country(token)

			features.word_shape = utils.get_word_shape(token)
		return features

	@staticmethod
	def proc_lines(lines, location):
		line_stat = "LINESTART"
		result = []
		location = location[0]
		cur_pos_place = 0
		cur_token = 0

		for i, line in enumerate(lines):
			is_loc = False
			if line == "\n":
				result.append("\n \n")
				continue

			for j in range(cur_pos_place, len(location)):
				loc = location[j]
				if loc[0] <= cur_token and loc[1] >= cur_token:
					is_loc = True
					cur_pos_place = j
					break
				elif loc[0] > cur_token:
					is_loc = False
					cur_pos_place = j
					break


			if (line.strip() == "@newline"):
				line_stat = "LINESTART"
				continue

			if line.strip() == "":
				result.append("\n")
				line_stat = "LINESTART"
			else:
				if (i+1) < len(lines):
					if lines[i+1].strip() == "":
						line_stat = "LINEEND"
				elif (i+1) == len(lines):
					line_stat = "LINEEND"

				vector = FeatureVectorAffiliation.proc_line(line, line_stat, is_loc)
				result.append(vector.gen_vector())

				if line_stat == "LINESTART":
					line_stat = "LINEIN"

			cur_token += 1

		return result



	def gen_vector(self):
		if self.string is None:
			return None
		if len(self.string) == 0:
			return None
		res = [self.string]
		s = self.string
		res.append(s.lower())
		# prefix
		res.append(s[:1])
		res.append(s[:2])
		res.append(s[:3])
		res.append(s[:4])
		# suffix
		res.append(s[-1])
		res.append(s[-2:])
		res.append(s[-3:])
		res.append(s[-4:])

		res.append(self.line_stat)
		if self.digit == "ALLDIGIT":
			res.append("NOCAPS")
		else:
			res.append(self.capital)
		res.append(self.digit)

		if self.single_char:
			res.append("1")
		else:
			res.append("0")

		if self.proper_name:
			res.append("1")
		else:
			res.append("0")

		if self.common_name:
			res.append("1")	
		else:
			res.append("0")

		if self.location_name:
			res.append("1")
		else:
			res.append("0")

		if self.country_name:
			res.append("1")
		else:
			res.append("0")

		res.append(self.punct_type)

		res.append(self.word_shape)

		if self.label is not None:
			res.append(self.label)
		else:
			res.append("0")
		return " ".join(res)