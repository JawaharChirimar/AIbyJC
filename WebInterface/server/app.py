#!/usr/bin/env python3
"""
Flask server for digit extraction and classification web interface.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add the DigitNN directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "DigitNN"))

from DigitsExtractor import process_image

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Base directory for models
MODELS_BASE_DIR = Path(__file__).parent.parent.parent / "DigitNN" / "data" / "modelForDE"
IMAGE_SIZE = 64

def process_image_for_api(image_array, classifier_model_path):
    """
    Process image and return results as list of dicts.
    
    Args:
        image_array: numpy array of the image (BGR format)
        classifier_model_path: Path to classifier model (.keras file)
    
    Returns:
        dict with 'error' key OR 'results' key containing list of dicts:
        - Each dict: {'image': base64_encoded_image, 'digit': int, 'confidence': float}
    """
    try:
        # image_array is given hence input_path will be ignored.
        # set input_path to None.
        # return_results is True hence output_dir will be ignored.
        # set output_dir to None.
        # set input_size to 64 since in frontend 
        # we are using 64x64 images ONLY
        result = process_image(
            input_path=None,
            output_dir=None,
            classifier_model_path=classifier_model_path,
            classify_digits=True,
            image_array=image_array,
            return_results=True,
            input_size=IMAGE_SIZE
        )
        return result
    except Exception as e:
        return {'error': f'Processing error: {str(e)}'}


@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available model files."""
    try:
        models = []
        if MODELS_BASE_DIR.exists():
            # Find all .keras files
            for model_file in MODELS_BASE_DIR.rglob('*.keras'):
                relative_path = model_file.relative_to(MODELS_BASE_DIR.parent.parent.parent)
                models.append({
                    'path': str(model_file),
                    'name': model_file.name,
                    'run': model_file.parent.name if model_file.parent.name.startswith('run_') else 'unknown'
                })
        
        # Sort by path (most recent first if sorted)
        models.sort(key=lambda x: x['path'], reverse=True)
        
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Classify digits in uploaded image."""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file size (2MB limit)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > 2 * 1024 * 1024:  # 2MB
            return jsonify({'error': 'File size exceeds 2MB limit'}), 400
        
        # Check if model path is provided
        model_path = request.form.get('model_path')
        if not model_path:
            return jsonify({'error': 'Model path not provided'}), 400
        
        # Check if model file exists
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found: {model_path}'}), 400
        
        # Read image
        file_content = file.read()
        nparr = np.frombuffer(file_content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Could not decode image. Please ensure it is a valid JPEG or PNG file.'}), 400
        
        # Process image
        result = process_image_for_api(image, model_path)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
