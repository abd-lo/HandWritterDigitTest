import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageOps

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "digit_model.keras"

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    raise FileNotFoundError("Please run train.py first!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        # Open image and convert to grayscale
        img = Image.open(file.stream).convert('L')

        # Convert to numpy array
        img_array = np.array(img)

        # If background is white, invert colors
        if np.mean(img_array) > 127:
            img = ImageOps.invert(img)

        # Convert back to array
        img_array = np.array(img)

        # Simple threshold to remove noise
        img_array = (img_array > 50).astype(np.uint8) * 255

        # Crop the digit area
        coords = np.argwhere(img_array > 0)

        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            img_array = img_array[y0:y1, x0:x1]

        # Convert back to PIL image
        img = Image.fromarray(img_array)

        # Resize while keeping aspect ratio
        img.thumbnail((20, 20))

        # Create a 28x28 black canvas
        canvas = Image.new('L', (28, 28), 0)

        # Center the digit
        x = (28 - img.width) // 2
        y = (28 - img.height) // 2
        canvas.paste(img, (x, y))

        # Normalize
        img_array = np.array(canvas).astype('float32') / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)

        digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)

        return jsonify({
            'digit': digit,
            'confidence': f'{confidence:.2f}%'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)