from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from joblib import load  # Load LabelEncoder

app = Flask(__name__)

# Configuration for paths
MODEL_PATH = "../models/IRISRecognizer.h5"
LABEL_ENCODER_PATH = 'label_encoder.joblib'
UPLOAD_FOLDER = 'uploads'

# Load the Keras model for human iris detection
model = load_model(MODEL_PATH)

# Load LabelEncoder for decoding predictions
label_encoder = load(LABEL_ENCODER_PATH)

# Function to prepare the image for prediction


def prepare_image(image_path):
    image = load_img(image_path, target_size=(
        150, 150), color_mode='grayscale')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


# Ensure 'uploads' directory exists for saving uploaded images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for the home page


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/predict')
def show_predict_page():
    return render_template('predict.html')



@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        image = prepare_image(file_path)

        predict = model.predict(image)
        label = np.argmax(predict, axis=1)[0]
        predict_label = label_encoder.inverse_transform([label])[0]

        return jsonify({'label': predict_label})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
