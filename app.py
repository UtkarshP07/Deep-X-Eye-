import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)
app.secret_key = "abc"
app.config['UPLOAD_FOLDER'] = "User_Images"

# Make sure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your model
model = load_model("deepxeyefinal.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        f = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = ['No Diabetic Retinopathy', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']
        result = index[prediction[0]]
        return jsonify({'prediction': result})

if __name__ == "__main__":
    app.debug = True
    app.run()
