import utils

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

	@static_method
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

			if utils.test_all_digit(token):
				features.digit = "ALLDIGIT"
			elif utils.test_contain_digit(token):
				features.digit = "CONTAINDIGIT"
			else:
				features.digit = "NODIGIT"

			features.punct_type = "NOPUNCT"





	def add_feature(lines, location):
		line_stat = "LINESTART"
		result = []

		for line in lines:
			is_loc = False
			if line == "\n":
				result.append("\n \n")
				continue
			skip_test = False
			for 



	def gen_vector():
		if self.string is None:
			return None
		if len(self.string) = 0:
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

