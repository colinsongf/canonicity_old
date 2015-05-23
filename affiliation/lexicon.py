from matcher import Matcher

class Lexicon():
	def __init__(self):
		self.city_matcher = None

	def init_cities(self):
		self.city_matcher = Matcher()
		self.city_matcher.load_terms("../lexicon/places/cities15000.txt")

	def in_city_names(self, tokens):
		if self.city_matcher is None:
			self.init_cities()
		return self.city_matcher.match(tokens)