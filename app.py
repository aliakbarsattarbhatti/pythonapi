from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the model
model_path = r"content"
model = tf.saved_model.load(model_path)


def preprocess_image(image):
    # Preprocess the image
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = image / 255.0
    return image


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read the image file
        image = Image.open(file)
        # Preprocess the image
        processed_image = preprocess_image(image)
        # Perform inference
        output = model(processed_image)
        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(output)
        # Get the predicted class indices
        predicted_class_indices = tf.argsort(probabilities, axis=1, direction='DESCENDING').numpy()[0]
        # Get the corresponding probabilities
        confidence_scores = probabilities.numpy()[0][predicted_class_indices]

        # Prepare response data
        response_data = []
        for class_index, confidence in zip(predicted_class_indices, confidence_scores):
            response_data.append({'class_index': class_index.item(), 'confidence': confidence.item()})

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})


# if __name__ == "__main__":
#     app.run(debug= True)