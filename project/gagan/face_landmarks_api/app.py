import logging

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

import face_recognizition_api as api


app = Flask(__name__)


@app.route('/getkeypts', methods=['POST'])
def get_keypts():
    """
    Flask application that receives image and return the facial landmark points.
    """
    try:
        image = request.files['file']
        logging.info("Image recieved")
        image = np.array(Image.open(image))
        logging.info(str(type(image)))
        result = api.get_encoding(image)
        return result
    except Exception:
        return "Something went wrong", 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0') 
