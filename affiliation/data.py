class Affiliation(object):
	def __init__(self):
		self.acronym = None
		self.name = None
		self.url = None
		self.institutions = None
		self.department = None
		self.laboratories = None
		self.country = None
		self.post_code = None
		self.post_box = None
		self.region = None
		self.city = None
		self.addr_line = None
		self.marker = None
		self.address_string = None
		self.affiliation_string = None
		self.fail_affiliation = None

	def to_dict(self):
		pass