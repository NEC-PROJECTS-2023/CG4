from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH ='E:/COVID-19-Detection/Covid_cnn.h5'
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(100, 100))
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    print(preds)
    preds=np.argmax(preds, axis=1)
    print(preds)
    if preds==0:
        preds="THE IMAGE IS COVID POSITIVE"
    elif preds==1:
        preds="THE IMAGE IS NORMAL IMAGE"
    elif preds==2:
        preds="THE IMAGE CONTAINS VIRAL PNEUMONIA"
    else:
        preds="Not a chest x-ray image"
        
    
    
    return preds
@app.route('/', methods=['GET'])
def index():
    
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None
if __name__ == '__main__':
    app.run(port=5001,debug=True)
    
    
    
