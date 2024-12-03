from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import logging
import traceback


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

try:
    model = tf.keras.models.load_model(r'C:\Users\mutha\Desktop\Heart\heart_disease_autoencoder.h5')
except Exception as e:
    logging.error(f"Model loading error: {e}")
    model = None

def preprocess_image(image, target_size=(224, 224)):
    try:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image)
        
        if image_array.ndim == 2:
            image_array = np.stack((image_array,) * 3, axis=-1)
        
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logging.error(f"Image preprocessing error: {e}")
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate file
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})
   
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"})
        
        # Open and preprocess image
        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict
        prediction = model.predict(image_array)
        
        # For autoencoders, calculate reconstruction error
        reconstruction_error = np.mean(np.abs(prediction - image_array))
        
        # Threshold for heart disease detection
        label = "Heart Disease" if 0.20 < reconstruction_error < 0.30 else "No Heart Disease"
        
        return jsonify({
            "prediction": label,
            "confidence": float(reconstruction_error)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)