from matcher import Matcher
import utils
import logging

class Lexicon():
	def __init__(self):
		self.city_matcher = None
		self.countries = None
		self.country_codes = None

	def init_countries(self):
		self.countries, self.country_codes = utils.parse_country_codes("../lexicon/countries/CountryCodes.xml")
		logging.info("Country codes initialized")

	def init_cities(self):
		self.city_matcher = Matcher()
		self.city_matcher.load_terms("../lexicon/places/cities15000.txt")
		logging.info("City matcher initialized")

	def in_city_names(self, tokens):
		if self.city_matcher is None:
			self.init_cities()
		return self.city_matcher.match(tokens)

	def is_country(self, token):
		if self.countries is None:
			self.init_countries()
		return token.lower() in self.countries

lex = Lexicon()