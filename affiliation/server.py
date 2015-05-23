from flask import Flask
app = Flask(__name__)

from parser import AffilaitonParser
aff_parser = AffilaitonParser()

import json

@app.route("/parse/<query>")
def parse_affliation(query):
	print query
	result = aff_parser.process(query)
	print result
	return json.dumps(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)