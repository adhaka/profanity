from pprint import pprint

application_port = 1234

import web

import os, sys, json
from Profanity import Profanity as Pf

path = os.path.abspath(__file__)
source_code_folder = os.path.dirname(os.path.dirname(path))
sys.path.append(source_code_folder)

urls = (
    '/hello', 'hello'
)

def factorial(n):
	if n == 0 or n == 1:
			return 1
	else:
		return n* factorial(n-1)

class hello:
    def POST(self):
        query = web.data()

        query = json.loads(query)

        # pprint({'query': query})

        pd = query['pd']
        Profane = Pf(1)
        output = Profane.predictDescription(pd) 
        return output

        # return factorial(n)

if __name__ == "__main__":
    app = web.application(urls, globals())
    # app.run()
    web.httpserver.runsimple(app.wsgifunc(), ("0.0.0.0", application_port))