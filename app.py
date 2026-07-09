import os

# Disable GPU (important for free cloud hosting)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# Reduce TensorFlow memory usage
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageOps


app = Flask(__name__)

MODEL_PATH = "digit_model.keras"

# Model will be loaded only when needed
model = None


def load_model():
    global model

    if model is None:

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model file digit_model.keras was not found!"
            )

        print("Loading TensorFlow model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")

    return model



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400


    file = request.files["file"]


    try:

        # -----------------------------
        # Load image
        # -----------------------------

        img = Image.open(file.stream).convert("L")

        img_array = np.array(img)


        # -----------------------------
        # Convert background
        # MNIST uses white digit on black background
        # -----------------------------

        if np.mean(img_array) > 127:
            img = ImageOps.invert(img)


        img_array = np.array(img)


        # -----------------------------
        # Remove noise
        # -----------------------------

        img_array = (
            (img_array > 50)
            .astype(np.uint8)
            * 255
        )


        # -----------------------------
        # Crop digit
        # -----------------------------

        coords = np.argwhere(img_array > 0)


        if coords.size > 0:

            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1

            img_array = img_array[y0:y1, x0:x1]



        # -----------------------------
        # Resize while keeping ratio
        # -----------------------------

        img = Image.fromarray(img_array)

        img.thumbnail((20, 20))


        # Create MNIST style canvas
        canvas = Image.new(
            "L",
            (28, 28),
            0
        )


        x = (28 - img.width) // 2
        y = (28 - img.height) // 2


        canvas.paste(
            img,
            (x, y)
        )


        # -----------------------------
        # Normalize
        # -----------------------------

        img_array = (
            np.array(canvas)
            .astype("float32")
            / 255.0
        )


        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # -----------------------------
        # Prediction
        # -----------------------------

        trained_model = load_model()


        predictions = trained_model.predict(
            img_array,
            verbose=0
        )


        digit = int(
            np.argmax(predictions[0])
        )


        confidence = float(
            np.max(predictions[0]) * 100
        )


        return jsonify({

            "digit": digit,

            "confidence":
                f"{confidence:.2f}%"

        })


    except Exception as e:

        print("ERROR:", e)

        return jsonify({

            "error": str(e)

        }), 500




if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
