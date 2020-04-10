import flask
from flask import request, jsonify, Response

import json
import numpy as np
import olaip
import matplotlib.pyplot as plt
from skimage import io


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return '''<h1>OLASimple IP</h1>
<p>Python service providing the API for image processing of scanned OLASimple diagnostic strips.</p>
<p>Send a POST request to /api/processimage with the image to be processed.</p>'''


@app.route('/api/processimage', methods=['POST'])
def process():
    file = request.files['file']

    
    tstats = olaip.process_image_from_file(file, trimmed=False)    
    results = olaip.make_calls_from_tstats(tstats)
    # build a response dict to send back to client
    # response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
    #             }
    response = jsonify(results=results)

    return response


app.run()