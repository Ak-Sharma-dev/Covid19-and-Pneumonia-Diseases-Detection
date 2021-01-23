from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'Models/ML_model1.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        if preds[0][0] == 1 :
            prediction = "COVID-19"
        elif preds[0][1] == 1:
            prediction = "NORMAL"
        elif preds[0][2] == 1:
            prediction = "PNEUMONIA"
        return prediction

if __name__=='__main__':
    app.run(host='localhost', port='5002',debug=False)

