from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import io
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# GitHub direct download link
MODEL_URL = "https://github.com/yadhu-vipin/brainiac/releases/download/v1.0.0/brain_tumor_model.keras"
MODEL_PATH = "brain_tumor_model.keras"

# Check if model file already exists
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from GitHub...")
    response = requests.get(MODEL_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("✅ Model downloaded successfully!")
    else:
        print(f"❌ Failed to download model: {response.status_code}")
        exit(1)
else:
    print("✅ Model already exists, skipping download.")

# Load the model once
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Define class labels
CLASS_LABELS = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((224, 224))  # Resize to model input size

        img_array = np.array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class])

        return jsonify({
            "prediction": CLASS_LABELS[predicted_class], 
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
