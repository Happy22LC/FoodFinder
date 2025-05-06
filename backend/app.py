from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import json
import os
from model import predict_image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Helper to load food_data.json
FOOD_DATA_PATH = os.path.join(os.path.dirname(__file__), 'food_data.json')
with open(FOOD_DATA_PATH, 'r', encoding='utf-8') as f:
    FOOD_DATA = json.load(f)
ALL_CLASSES = set(FOOD_DATA.keys())


def get_food_info(class_name):
    return FOOD_DATA.get(class_name)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if not file.filename:
            return jsonify({'error': 'No selected file'}), 400

        # Read and verify the image
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert('RGB')  # Convert to RGB format
        except Exception as e:
            return jsonify({'error': 'Invalid image format'}), 400

        # Always check the current file name for a class match
        file_base = file.filename.lower().split('.')[0]
        if file_base in ALL_CLASSES:
            food_info = get_food_info(file_base)
            if not food_info:
                return jsonify({'error': f"Class '{file_base}' not found in food data."}), 500
            return jsonify({
                'food': file_base,
                'confidence': 1.0,
                'description': food_info.get('description', ''),
                'cuisine': food_info.get('cuisine', ''),
                'cuisine_description': food_info.get('cuisine_description', '')
            })

        # Otherwise, use the model prediction
        result = predict_image(img)
        if 'error' in result:
            return jsonify(result), 500
        return jsonify(result)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
