from flask import Flask
from flask import jsonify
from flask import request, render_template

import io
from io import BytesIO

from PIL import Image
import base64
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array

from datetime import datetime
import os
from google.cloud import storage
import tempfile

app = Flask(__name__)

@app.route('/')
def index():

    return render_template('web-interface.html')

os.environ["GCLOUD_PROJECT"] = "{ENTER YOUR CLOUD PROJECT}"
storage_client = storage.Client()
bucket_name = '{ENTER YOUR BUCKET NAME}'
model = 'model.h5'
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(model)

temp_path = "/tmp/model.h5"
blob.download_to_filename(temp_path)

def get_model():
    global model
    model = tf.keras.models.load_model(temp_path)
    print(" * Disney-Model_loaded!")
    return model

print(" * Loading Keras Model")
get_model()

@app.route("/predict", methods = ["POST"])
def predict():
    message = request.get_json(force=True)
    print("Message: ", message)
    encoded = message['image']

    encoded_bytes = str.encode(encoded)
    now = datetime.now()
    encoded += "=="
    decoded = base64.b64decode(encoded)
    
    #Convert decoded into Tensor
    image = tf.image.decode_image(decoded, channels=3)
    print("TF Image Decoded: ", image)
    print("Image Shape: ", image.shape )

    #Convert to Float
    image = tf.image.convert_image_dtype(image, tf.float32)
    print("TF Image Converted to Float: ", image)
    print("Image Shape: ", image.shape )

    #Resize Tensor
    image = tf.image.resize(image, [224, 224])
    print("TF Image Resize: ", image)
    print("Image Shape: ", image.shape )

    #Expanding (1, 224, 224, 3)
    processed_image = np.expand_dims(image, axis=0)
    print("TF Image Resize: ", processed_image)
    print("Image Shape: ", processed_image.shape )

    prediction  =  model.predict(processed_image).tolist()

    response = {'prediction': {
            'anna':prediction[0][0],
            'ariel':prediction[0][1],
            'aurora':prediction[0][2],
            'belle':prediction[0][3],
            'cinderella':prediction[0][4],
            'elsa':prediction[0][5],
            'jasmine':prediction[0][6],
            'merida':prediction[0][7],
            'moana':prediction[0][8],
            'mulan':prediction[0][9],
            'pocahontas':prediction[0][10],
            'rapunzel':prediction[0][11],
            'snow':prediction[0][12],
            'tiana':prediction[0][13]

    }              }
    print(response)
    return jsonify(response)
