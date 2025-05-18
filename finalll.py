import os
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.saving import register_keras_serializable
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load MobileNetV2 model
mobilenet_model = MobileNetV2(weights="imagenet")
mobilenet_model = tf.keras.Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Register LSTM for custom objects
@register_keras_serializable()
class CustomLSTM(tf.keras.layers.LSTM):
    pass

# Load the trained LSTM model
model = tf.keras.models.load_model('mymodel.h5', custom_objects={'LSTM': CustomLSTM}, compile=False)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Caption generation function
def get_word_from_index(index, tokenizer):
    return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)

def predict_caption(model, image_features, tokenizer, max_caption_length=34):
    caption = "startseq"
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)
        if predicted_word is None or predicted_word == "endseq":
            break
        caption += " " + predicted_word
    return caption.replace("startseq", "").replace("endseq", "").strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Process image
    image = load_img(filepath, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Extract features
    image_features = mobilenet_model.predict(image, verbose=0)

    # Generate caption
    caption = predict_caption(model, image_features, tokenizer)

    return jsonify({"caption": caption, "image_url": filepath})

if __name__ == "__main__":
    app.run()