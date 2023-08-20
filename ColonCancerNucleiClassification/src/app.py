import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D,Conv2D,MaxPooling2D,BatchNormalization,Concatenate
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import os
import io
import re
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as image 

from keras.models import load_model
import keras.metrics
from keras.utils import custom_object_scope


from keras import backend as K

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return true_negatives / (true_negatives+false_positives + K.epsilon())

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)



app = Flask(__name__, template_folder='templates')
with custom_object_scope({'f1': f1,'sensitivity':sensitivity,'sorted_alphanumeric':sorted_alphanumeric,'specificity':specificity}):
    model = load_model("c:/Users/Santhosh Reddy/Downloads/cancermodel.h5")

# Define the target class labels
class_labels = ['Epithelial', 'Fibroblast', 'Inflammatory', 'Others']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # # Get the uploaded image file from the form
    image_file = request.files['image']
    # image_path = 'static/uploads/temp.jpeg'  # Assuming the static folder exists and has appropriate permissions
    # image_file.save(image_path)


    # Save the file to a temporary location
    # Read the file as bytes
    image_bytes = image_file.read()
    # Create a byte stream
    image_stream = io.BytesIO(image_bytes)
    # Load the image using Keras image module
    img = image.load_img(image_stream, target_size=(32, 32))

    # Preprocess the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img= preprocess_input(img)

    # Perform classification using the loaded model
    prediction = model.predict(img)
    prob=max(prediction[0])
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]


    # Render the result page with the predicted label
    return render_template('index.html', value='The predicted class is {} and its probability is {}'.format(predicted_label,prob))


if __name__ == "__main__":
    app.run(debug=True)
