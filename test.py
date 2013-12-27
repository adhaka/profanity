import urllib2
import json


if __name__ == '__main__':
	url = 'http://localhost:1234/hello'

	query = {}
	query['pd'] = "Very spacious apartment in the heart of mumbai. Affordable 2 bhk. This property is only for muslims"

	jsondata = json.dumps(query)
	response = urllib2.urlopen(url, jsondata)

	# jsonresp = response.read()

	# print jsonresp