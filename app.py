from flask import Flask, render_template, request
import tensorflow as tf
import keras
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the CNN model
model = keras.models.load_model('fruit_classifier.h5')

# Define the classes
classes = ['rottentomato', 'rottenpotato', 'rottenorange', 'rottenbanana', 'rottenapple', 
           'freshtomato', 'freshpotato', 'freshnorange', 'freshbanana', 'freshapple']

# Define the preprocessing function for the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']
        if image_file:
            # Save the uploaded image
            image_path = "static/uploads/" + image_file.filename
            image_file.save(image_path)
            # Preprocess the image
            img_array = preprocess_image(image_path)
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_class = classes[predicted_class_index]
            if predicted_class.startswith('rotten'):
                predicted_class = predicted_class[len('rotten'):]
                freshness = 'Rotten'
            elif predicted_class.startswith('fresh'):
                predicted_class = predicted_class[len('fresh'):]
                freshness = 'Fresh'
            else:
                predicted_class = predicted_class[:5]
                freshness = 'Unknown'
            predicted_class = predicted_class.capitalize()
            return render_template('result.html', fruit=predicted_class,image_filename=image_file.filename,freshness=freshness)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
