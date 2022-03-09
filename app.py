from __future__ import division, print_function

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('model.h5')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x/255
    x = np.expand_dims(x, axis=0)
   
    preds = model.predict(x)
    preds= np.argmax(preds, axis=1)
    if preds==0:
        preds="Back Side of Traffic Light"
    elif preds==1:
        preds="Green Light"
    elif preds==2:
        preds="Red Light"
    elif preds==3:
        preds = "Yellow Light"    
    
    
    return preds

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return render_template('index.html',prediction_text = result)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)    