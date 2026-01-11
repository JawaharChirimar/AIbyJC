# Digit Classifier Web Interface

A web interface for classifying handwritten digits in images using trained CNN models.

## Features

- Upload JPEG images (max 2MB) with multiple handwritten digits
- Select from available trained models
- View classification results with confidence scores
- Scrollable results display with digit images, predictions, and confidence values

## Setup

### Server Setup

1. Install dependencies:
```bash
cd server
pip install -r requirements.txt
```

2. Run the Flask server:
```bash
python app.py
```

The server will start on `http://localhost:5001`

### Frontend Setup

The frontend is static HTML/CSS/JavaScript. Simply open `frontend/index.html` in a web browser, or serve it using a simple HTTP server:

```bash
cd frontend
python -m http.server 5002
```

Then open `http://localhost:5002` in your browser.

**Note:** For development, you may need to serve the frontend from a web server due to CORS restrictions when loading from `file://`. The Flask server has CORS enabled, so API calls should work when served via HTTP.

## Usage

1. Open the frontend in a web browser
2. Select a model from the dropdown (models are loaded from `~/Development/AIbyJC/DigitNN/data/modelForDE/`)
3. Upload a JPEG image file (max 2MB) containing handwritten digits
4. Click "Classify Digits"
5. View the results in the scrollable results area

## API Endpoints

### GET `/api/models`
Returns a list of available model files.

Response:
```json
{
  "models": [
    {
      "path": "/full/path/to/model.keras",
      "name": "digit_classifier_epoch_01.keras",
      "run": "run_2026_01_06_17_12_48"
    }
  ]
}
```

### POST `/api/classify`
Processes an uploaded image and returns classification results.

Request:
- `file`: JPEG image file (multipart/form-data)
- `model_path`: Full path to the model file (form field)

Response (success):
```json
{
  "results": [
    {
      "image": "base64_encoded_jpeg_image",
      "digit": 5,
      "confidence": 0.98
    }
  ]
}
```

Response (error):
```json
{
  "error": "Error message"
}
```

## Technical Details

- Images are processed using contour detection to extract individual digits
- Each digit is normalized to 42x42 pixels grayscale
- The classifier model expects 28x28 images (resizing happens internally)
- Results include the processed 42x42 digit image, predicted digit (0-9), and confidence score (0-1)
