# Import the necessary libraries
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import base64

# Create a new Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'

# Load in the full model
model_path = './2023020310231675419784-full-image-set-mobilenetv2-Adam.h5'
loaded_full_model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})

# Define the image size
IMG_SIZE = 224


# Add the process_image function
def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image


# Add the get_image_label function
def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label


# Create the index page with the image upload form
@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Create the Flask endpoint for uploading an image and returning the predicted label and breed name.
@app.route('/predict', methods=['POST'], endpoint='predict_image')
def predict_image():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Check if the file is a JPG image and convert it to PNG format
    if file and file.filename.endswith('.jpg'):
        image = Image.open(file)
        file_path = file_path.replace('.jpg', '.png')
        image.save(file_path)
    elif file and file.filename.endswith('.png'):
        file.save(file_path)
    else:
        return render_template("index.html", error="Invalid file format. Please upload a JPG or PNG image.")

    processed_image = process_image(file_path)

    # Predict the label and breed name
    prediction = loaded_full_model.predict(tf.expand_dims(processed_image, axis=0))
    pred_label = np.argmax(prediction)
    breed_name = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier']

    # Get the breed name
    label = str(pred_label)
    breed = breed_name[int(label)]

    # Convert the processed image to base64 and pass it to the result template
    img_io = io.BytesIO()
    Image.fromarray((processed_image.numpy() * 255).astype(np.uint8)).resize((224, 224)).save(img_io, 'PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode('ascii')

    return render_template("result.html", predicted_label=label, breed_name=breed, image=img_base64)


# Run the Flask server on your localhost
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)