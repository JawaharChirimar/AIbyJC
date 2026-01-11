================================================================================
README for Digit Classification and Detection System
================================================================================

This document describes how to use the digit classification and detection 
system, which consists of two main Python scripts:

1. DigitClassifier.py - Trains and provides digit classification models
2. BoundingBoxFromYolo.py - Detects and extracts digits from images

================================================================================
INSTALLATION
================================================================================

1. Install Python dependencies from requirements file:

   pip install -r requirements_BoundingBoxFromYolo.txt

   This will install:
   - ultralytics>=8.0.0      (YOLO object detection)
   - opencv-python>=4.5.0    (Image processing)
   - numpy>=1.21.0           (Numerical operations)
   - tensorflow>=2.10.0      (Deep learning framework)
   - albumentations>=1.3.0   (Data augmentation)
   - emnist                   (Extended MNIST dataset)

   Note: TensorFlow and other packages may take several minutes to install.
         Make sure you have a stable internet connection.

2. Virtual Environment (Recommended):

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements_BoundingBoxFromYolo.txt

================================================================================
SCRIPT 1: DigitClassifier.py
================================================================================

WHAT IT DOES:
-------------
DigitClassifier.py provides functions to create, train, and use a CNN-based 
digit classifier that can recognize handwritten digits (0-9). The classifier 
is trained on the MNIST and/or EMNIST Digits datasets.

Key Features:
- Creates and trains a CNN model for digit classification
- Supports data augmentation during training
- Can use MNIST, EMNIST Digits, or both datasets
- Supports deep or shallow model architectures
- Saves models in TensorFlow/Keras format (.keras files)
- Saves checkpoint models after each training epoch

HOW TO RUN:
-----------

Standalone Training (Main Function):
------------------------------------
python DigitClassifier.py [OPTIONS]

Example:
  python DigitClassifier.py
  python DigitClassifier.py --epoch-count 30 --no-augment
  python DigitClassifier.py -m ./models/my_model.keras --force-retrain

All Command-Line Flags:
-----------------------

1. -m, --model-path PATH
   Description: Path to save/load the trained model (.keras file)
   Default: ~/.digit_classifier_mnist.keras
   Example: python DigitClassifier.py -m ./my_model.keras

2. --force-retrain
   Description: Force retraining even if a model already exists
   Default: False (loads existing model if found)
   Example: python DigitClassifier.py --force-retrain

3. --no-augment
   Description: Disable data augmentation during training
   Default: Augmentation is ENABLED (True)
   Example: python DigitClassifier.py --no-augment

4. --no-mnist
   Description: Exclude MNIST data from training/validation
   Default: MNIST is INCLUDED (True)
   Example: python DigitClassifier.py --no-mnist

5. --no-emnist
   Description: Exclude EMNIST Digits data from training/validation
   Default: EMNIST is INCLUDED (True)
   Example: python DigitClassifier.py --no-emnist

6. --epoch-count N
   Description: Number of training epochs
   Default: 20
   Example: python DigitClassifier.py --epoch-count 30

7. --no-deep-model
   Description: Use shallow model architecture instead of deep model
   Default: Deep model is USED (True)
   Example: python DigitClassifier.py --no-deep-model

Combined Examples:
------------------
# Default training (MNIST + EMNIST, augmentation, deep model, 20 epochs)
python DigitClassifier.py

# Train with shallow model, 30 epochs, no augmentation
python DigitClassifier.py --no-deep-model --epoch-count 30 --no-augment

# Train with MNIST only, shallow model, custom path
python DigitClassifier.py --no-emnist --no-deep-model -m ./models/my_model.keras

# Force retrain with EMNIST only, deep model, 50 epochs
python DigitClassifier.py --force-retrain --no-mnist --epoch-count 50

# Minimal training: MNIST only, shallow model, no augmentation, 10 epochs
python DigitClassifier.py --no-emnist --no-deep-model --no-augment --epoch-count 10

FUNCTION: load_or_create_digit_classifier()
--------------------------------------------
WHAT IT DOES:
  Loads an existing trained model, or creates and trains a new digit 
  classifier model.

HOW TO USE (Python function):
  from DigitClassifier import load_or_create_digit_classifier
  
  model = load_or_create_digit_classifier(
      classifier_model_path=None,
      use_augmentation=True,
      use_mnist=True,
      use_emnist=True,
      num_epochs=20,
      use_deep_model=True
  )

PARAMETERS:
  classifier_model_path: str, optional
      Path to saved classifier model (.keras file)
      Default: None (uses ~/.digit_classifier_mnist.keras)
      If model exists at this path, it will be loaded instead of training.
      
  use_augmentation: bool, optional
      Whether to use data augmentation during training
      Default: True
      When True, applies rotation, slant, blur, noise, and stroke thickness
      variations to training data. Only 50% of samples are augmented.
      
  use_mnist: bool, optional
      Whether to include MNIST data for training/validation
      Default: True
      MNIST provides 60,000 training and 10,000 test samples.
      
  use_emnist: bool, optional
      Whether to include EMNIST Digits data for training/validation
      Default: True
      EMNIST Digits provides ~240,000 training and ~40,000 test samples.
      Requires 'emnist' package to be installed.
      
  num_epochs: int, optional
      Number of training epochs
      Default: 20
      Each epoch processes all training samples once.
      
  use_deep_model: bool, optional
      Whether to use deep model architecture
      Default: True
      - True: 4 Conv2D layers (more parameters, better for large datasets)
      - False: 2 Conv2D layers (fewer parameters, faster training)
      Model capacity also adjusts based on use_emnist:
        * With EMNIST: conv4=64, dense1=128 (deep) or conv4=32, dense1=32 (shallow)
        * MNIST only: conv4=32, dense1=32 (both architectures)

RETURNS:
  A trained Keras model that can classify 28x28 greyscale digit images.

OUTPUT FILES:
  When training a new model, the following files are created:
  - ~/data/modelForBBFY/run_YYYY_MM_DD_HH_MM_SS/digit_classifier_epoch_XX.keras
    (One model per epoch)
  - ~/data/modelForBBFY/run_YYYY_MM_DD_HH_MM_SS/digit_classifier_final.keras
    (Final trained model)
  - If model_path is specified: model is also saved to that location

FUNCTION: classify_digit()
---------------------------
WHAT IT DOES:
  Classifies a single 28x28 greyscale digit image using a trained model.

HOW TO USE (Python function):
  from DigitClassifier import classify_digit
  
  predicted_digit, confidence = classify_digit(classifier_model, digit_image)

PARAMETERS:
  classifier_model: Keras model
      A trained digit classifier model (returned by load_or_create_digit_classifier)
      Required parameter.
      
  digit_image: numpy array, shape (28, 28)
      A 28x28 greyscale image of a digit
      Expected format: White digits on black background (MNIST format)
      Pixel values: 0-255 (will be normalized internally to 0-1)
      If image is not 28x28, it will be resized automatically.

RETURNS:
  predicted_digit: int (0-9)
      The predicted digit class
      
  confidence: float (0.0-1.0)
      The model's confidence in the prediction

Example:
  import cv2
  from DigitClassifier import load_or_create_digit_classifier, classify_digit
  
  # Load model
  model = load_or_create_digit_classifier()
  
  # Load and classify a digit image
  img = cv2.imread('digit.jpg', cv2.IMREAD_GRAYSCALE)
  digit, conf = classify_digit(model, img)
  print(f"Predicted: {digit} (confidence: {conf:.2%})")

FUNCTION: create_digit_classifier_model()
------------------------------------------
WHAT IT DOES:
  Creates (but does not train) a CNN model for digit classification.

HOW TO USE (Python function):
  from DigitClassifier import create_digit_classifier_model
  
  model = create_digit_classifier_model(use_emnist=True, use_deep_model=True)

PARAMETERS:
  use_emnist: bool, optional
      Whether EMNIST data will be used (affects model capacity)
      Default: True
      Affects the number of filters/neurons in the model.
      
  use_deep_model: bool, optional
      Whether to use deep model architecture
      Default: True
      - True: 4 Conv2D layers
      - False: 2 Conv2D layers

RETURNS:
  A compiled but untrained Keras Sequential model ready for training.

MODEL ARCHITECTURES:
--------------------
Deep Model (use_deep_model=True):
  Input(28, 28, 1)
    -> Conv2D(conv4) -> BatchNorm -> Conv2D(conv4) -> BatchNorm
    -> MaxPool2D -> Dropout(0.25)
    -> Conv2D(conv4) -> BatchNorm -> Conv2D(conv4) -> BatchNorm
    -> MaxPool2D -> Dropout(0.25)
    -> Flatten
    -> Dense(dense1) -> BatchNorm -> Dropout(0.5)
    -> Dense(10, softmax)

Shallow Model (use_deep_model=False):
  Input(28, 28, 1)
    -> Conv2D(conv4) -> BatchNorm
    -> MaxPool2D -> Dropout(0.25)
    -> Conv2D(conv4) -> BatchNorm
    -> MaxPool2D -> Dropout(0.25)
    -> Flatten
    -> Dense(dense1) -> BatchNorm -> Dropout(0.5)
    -> Dense(10, softmax)

Where:
  - conv4 = 64, dense1 = 128 (when use_emnist=True)
  - conv4 = 32, dense1 = 32 (when use_emnist=False)

================================================================================
SCRIPT 2: BoundingBoxFromYolo.py
================================================================================

WHAT IT DOES:
-------------
BoundingBoxFromYolo.py processes a JPEG image containing handwritten digits.
It uses YOLO object detection (with contour-based fallback) to detect digit 
regions, extracts each region, normalizes them to 28x28 greyscale images, and 
saves them as individual JPEG files. Optionally, it can classify each digit 
using the trained CNN model.

Key Features:
- Detects digit regions using YOLO or contour detection
- Extracts and processes each digit region
- Normalizes to 28x28 greyscale (white digits on black background)
- Saves individual digit images with naming: file_L_D.jpg
  (L = line number, 0-indexed; D = digit number, 1-indexed)
- Optional digit classification using the CNN model
- Sorts digits in reading order (top-to-bottom, left-to-right)

HOW TO RUN:
-----------

Basic Usage:
------------
python BoundingBoxFromYolo.py INPUT_IMAGE [OPTIONS]

Example:
  python BoundingBoxFromYolo.py image.jpg
  python BoundingBoxFromYolo.py digits.png -o ./output --classify
  python BoundingBoxFromYolo.py scan.jpg -m custom_yolo.pt -c

Command-Line Arguments:
-----------------------

1. input_image (positional, required)
   Description: Path to input JPEG/PNG file with handwritten digits
   Example: python BoundingBoxFromYolo.py my_digits.jpg

2. -m, --model PATH
   Description: Path to YOLO model file (optional)
   Default: None (uses default YOLOv8n model)
   Example: python BoundingBoxFromYolo.py image.jpg -m yolov8n.pt
   Note: The default YOLOv8n model is trained on COCO dataset and will NOT
         detect handwritten digits. It will fall back to contour detection.
         For best results, use a YOLO model trained specifically for digits.

3. -o, --output PATH
   Description: Output directory for extracted digit images
   Default: None (creates timestamped directory ~/data/BBFY/run_YYYY_MM_DD_HH_MM_SS/)
   Example: python BoundingBoxFromYolo.py image.jpg -o ./my_output

4. -c, --classify
   Description: Enable digit classification using CNN
   Default: False (only extracts regions, no classification)
   Example: python BoundingBoxFromYolo.py image.jpg -c
   Note: If no classifier model exists, will automatically train one on MNIST
         (this may take several minutes on first run).

5. --classifier-model PATH
   Description: Path to pre-trained digit classifier model (.keras file)
   Default: None (uses default or trains new model)
   Example: python BoundingBoxFromYolo.py image.jpg -c --classifier-model ./my_model.keras
   Note: Only used when -c/--classify is enabled.

Complete Example:
-----------------
python BoundingBoxFromYolo.py \
    handwritten_digits.jpg \
    -m custom_digit_detector.pt \
    -o ./extracted_digits \
    -c \
    --classifier-model ./trained_model.keras

FUNCTION: process_image()
--------------------------
WHAT IT DOES:
  Main processing function that detects digits, extracts regions, processes 
  them, and saves individual digit images. Optionally classifies each digit.

HOW TO USE (Python function):
  from BoundingBoxFromYolo import process_image
  
  process_image(
      input_path,
      model_path=None,
      output_dir=None,
      classifier_model_path=None,
      classify_digits=False
  )

PARAMETERS:
  input_path: str, required
      Path to input image file (JPEG, PNG, etc.)
      Must be a valid image file path.
      
  model_path: str, optional
      Path to YOLO model file (.pt file)
      Default: None (uses default YOLOv8n model)
      The default model is not trained on digits, so contour detection 
      will be used as fallback.
      
  output_dir: str, optional
      Output directory for extracted digit images
      Default: None (creates timestamped directory)
      Format: ~/data/BBFY/run_YYYY_MM_DD_HH_MM_SS/
      
  classifier_model_path: str, optional
      Path to pre-trained digit classifier model (.keras file)
      Default: None
      Only used if classify_digits=True.
      If None and classify_digits=True, will train a new model automatically.
      
  classify_digits: bool, optional
      Whether to classify each extracted digit
      Default: False
      When True, uses the CNN classifier to predict the digit class and
      confidence score. Results are included in the output filename.

OUTPUT:
  Creates individual JPEG files for each detected digit:
  - Without classification: file_L_D.jpg
    Example: file_0_1.jpg, file_0_2.jpg, file_1_1.jpg
  - With classification: file_L_D_classified_X_conf_Y.jpg
    Example: file_0_1_classified_5_conf_0.95.jpg

  Where:
    L = Line number (0-indexed, top to bottom)
    D = Digit number within line (1-indexed, left to right)
    X = Predicted digit (0-9)
    Y = Confidence score (0.00-1.00)

DIGIT PROCESSING PIPELINE:
---------------------------
1. Image Detection:
   - First tries YOLO object detection
   - Falls back to contour detection if YOLO finds nothing
   - Filters by area and aspect ratio to find digit-like regions

2. Sorting:
   - Groups detections into lines based on y-coordinate
   - Sorts lines top-to-bottom
   - Sorts digits within each line left-to-right

3. Extraction and Processing:
   - Extracts each digit region from the original image
   - Converts to greyscale
   - Resizes to 20x20 pixels (preserving aspect ratio)
   - Applies noise reduction (bilateral filter)
   - Applies adaptive thresholding for contrast
   - Ensures white digits on black background (MNIST format)
   - Adds 4 pixels of black padding on all sides (final: 28x28)

4. Classification (if enabled):
   - Loads or trains digit classifier model
   - Classifies each extracted 28x28 image
   - Adds prediction and confidence to filename

FUNCTION: detect_digits_with_contours()
----------------------------------------
WHAT IT DOES:
  Fallback method using OpenCV contour detection to find digit regions when 
  YOLO doesn't detect anything.

HOW TO USE (Python function):
  from BoundingBoxFromYolo import detect_digits_with_contours
  
  detections = detect_digits_with_contours(
      image,
      min_area=50,
      max_area=None,
      aspect_ratio_range=(0.2, 3.0)
  )

PARAMETERS:
  image: numpy array
      Input image (BGR format from cv2)
      
  min_area: int, optional
      Minimum contour area to consider (filters out noise)
      Default: 50
      
  max_area: int or None, optional
      Maximum contour area (None = 10% of image area)
      Default: None
      
  aspect_ratio_range: tuple, optional
      (min, max) aspect ratio for bounding boxes
      Default: (0.2, 3.0)
      Filters out regions that are too wide or too tall

RETURNS:
  List of detection boxes: [(x1, y1, x2, y2), ...]

FUNCTION: extract_and_process_region()
---------------------------------------
WHAT IT DOES:
  Extracts a region from an image based on a bounding box, processes it to 
  match MNIST format (28x28, white digits on black background), and returns 
  the processed image.

HOW TO USE (Python function):
  from BoundingBoxFromYolo import extract_and_process_region
  
  processed = extract_and_process_region(image, box, target_size=(28, 28))

PARAMETERS:
  image: numpy array
      Input image (BGR format from cv2)
      
  box: tuple
      Bounding box (x1, y1, x2, y2)
      
  target_size: tuple, optional
      Target size (width, height)
      Default: (28, 28)

RETURNS:
  Processed 28x28 greyscale image (numpy array, uint8, 0-255)

================================================================================
WORKFLOW EXAMPLES
================================================================================

Example 1: Train a digit classifier
------------------------------------
# Train with default settings (MNIST + EMNIST, augmentation, deep model)
python DigitClassifier.py

# Train with custom settings
python DigitClassifier.py \
    --epoch-count 30 \
    --no-deep-model \
    -m ./my_digit_classifier.keras

Example 2: Extract digits from image (no classification)
--------------------------------------------------------
python BoundingBoxFromYolo.py handwritten_digits.jpg

# Output: ~/data/BBFY/run_YYYY_MM_DD_HH_MM_SS/file_L_D.jpg

Example 3: Extract and classify digits
---------------------------------------
# Use existing classifier model
python BoundingBoxFromYolo.py image.jpg \
    -c \
    --classifier-model ~/.digit_classifier_mnist.keras

# Train classifier automatically if not found
python BoundingBoxFromYolo.py image.jpg -c

# Output: ~/data/BBFY/run_YYYY_MM_DD_HH_MM_SS/file_L_D_classified_X_conf_Y.jpg

Example 4: Complete pipeline
-----------------------------
# Step 1: Train classifier
python DigitClassifier.py \
    --epoch-count 20 \
    -m ./models/digit_classifier.keras

# Step 2: Extract and classify digits
python BoundingBoxFromYolo.py scan.jpg \
    -o ./results \
    -c \
    --classifier-model ./models/digit_classifier.keras

================================================================================
TROUBLESHOOTING
================================================================================

Issue: "No module named 'tensorflow'"
Solution: Install dependencies: pip install -r requirements_BoundingBoxFromYolo.txt

Issue: "No module named 'emnist'"
Solution: pip install emnist

Issue: "YOLO finds no detections"
Solution: The default YOLOv8n model is not trained on digits. The script will
          automatically use contour detection as fallback. For better results,
          train or obtain a YOLO model specifically for digit detection.

Issue: "Classification accuracy is poor"
Solution: 
  - Ensure extracted images are clear and have good contrast
  - Retrain the classifier with more epochs: --epoch-count 30
  - Use both MNIST and EMNIST: (default, or ensure --no-emnist is not used)
  - Use deep model: (default, or ensure --no-deep-model is not used)
  - Enable augmentation: (default, or ensure --no-augment is not used)

Issue: "EMNIST download fails"
Solution: 
  - Check internet connection
  - If you see "File is not a zip file" error, delete the corrupted cache:
    rm ~/.cache/emnist/emnist.zip
    Then retry.

Issue: "Model training is slow"
Solution:
  - Use shallow model: --no-deep-model
  - Reduce epochs: --epoch-count 10
  - Disable augmentation: --no-augment
  - Use only MNIST: --no-emnist

Issue: "Out of memory during training"
Solution:
  - Use shallow model: --no-deep-model
  - Use only MNIST: --no-emnist
  - Disable augmentation: --no-augment

================================================================================
FILE FORMATS AND LOCATIONS
================================================================================

Model Files:
  - Format: .keras (TensorFlow/Keras native format)
  - Default location: ~/.digit_classifier_mnist.keras
  - Epoch checkpoints: ~/data/modelForBBFY/run_YYYY_MM_DD_HH_MM_SS/

Output Images:
  - Format: JPEG (quality: 95%)
  - Location: ~/data/BBFY/run_YYYY_MM_DD_HH_MM_SS/
  - Naming: file_L_D.jpg or file_L_D_classified_X_conf_Y.jpg

Input Images:
  - Supported formats: JPEG, PNG (any format supported by OpenCV)
  - Recommended: High contrast, clear images with visible digits

================================================================================
ADDITIONAL NOTES
================================================================================

1. First Run:
   - YOLOv8n model will download automatically (~10MB)
   - MNIST dataset will download automatically (~11MB)
   - EMNIST dataset download takes longer (~2-3 minutes, ~600MB)
   - Training may take 10-30 minutes depending on hardware and settings

2. Model Architecture:
   - The model adapts based on dataset size (MNIST vs MNIST+EMNIST)
   - Deep model has more parameters and better accuracy but slower training
   - Shallow model trains faster but may have slightly lower accuracy

3. Data Augmentation:
   - Only 50% of training samples are augmented
   - Augmentations include: rotation (±48°), slant, blur, noise, stroke thickness
   - Applied on-the-fly during training (each epoch sees different variations)

4. Performance:
   - Typical accuracy on test set: 95-99%+ (depending on settings)
   - Training time: 10-30 minutes (CPU) or 2-5 minutes (GPU)
   - Inference time: <1ms per digit image

5. Compatibility:
   - Python 3.7+ recommended
   - Tested with TensorFlow 2.10+, but newer versions should work
   - Works on Linux, macOS, and Windows

================================================================================
END OF DOCUMENTATION
================================================================================

