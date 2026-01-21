#!/usr/bin/env python3
"""
DigitClassifierCNNSVM.py

CNN + SVM hybrid model that trains end-to-end.
Uses CNN for feature extraction and SVM loss (hinge loss) for classification.
"""

import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import albumentations as A

# Import dataset loaders
from GetArdis import load_ardis_dataset
from GetUSPS import load_usps_dataset
from DigitClassifierALL import create_negative_examples

try:
    from emnist import extract_training_samples, extract_test_samples
    EMNIST_AVAILABLE = True
except ImportError:
    EMNIST_AVAILABLE = False
    print("Warning: 'emnist' package not available. Install with: pip install emnist")

# =============================================================================
# CONFIGURABLE CONSTANTS
# =============================================================================
DROPOUT_RATE = 0.5
SAMPLE_RATIO = 1.00
AUGMENT_RATIO = 0.25
ROTATION_ANGLE = 30
SHEAR_ANGLE = 15


def digit_loss(y_true, y_pred):
    """
    Multi-class hinge loss for digit classification.
    Only applied to digit samples (where y_true >= 0).
    Negatives (y_true == -1) are ignored.
    """
    # Filter to only digit samples (where y_true >= 0)
    digit_mask = y_true >= 0
    if tf.reduce_any(digit_mask):
        digit_labels_filtered = tf.boolean_mask(y_true, digit_mask)
        digit_outputs_filtered = tf.boolean_mask(y_pred, digit_mask)
        return tf.keras.losses.sparse_categorical_hinge(digit_labels_filtered, digit_outputs_filtered)
    else:
        # No digits in this batch, return zero loss
        return 0.0


def binary_loss(y_true, y_pred):
    """
    Binary hinge loss for digit detection.
    y_true: 1 for digits, 0 for negatives
    y_pred: raw score (positive = digit, negative = not digit)
    """
    # Convert to shape (batch, 1) for binary hinge
    y_true = tf.expand_dims(tf.cast(y_true, tf.float32), axis=1)
    y_pred = tf.expand_dims(y_pred, axis=1)
    return tf.keras.losses.hinge(y_true, y_pred)


def svm_accuracy(y_true, y_pred):
    """
    Accuracy metric for SVM: predict class with highest score (digits only).
    """
    digit_labels, _ = y_true
    digit_outputs, _ = y_pred
    
    # Only evaluate on digit samples
    digit_mask = digit_labels >= 0
    if tf.reduce_any(digit_mask):
        digit_labels_filtered = tf.boolean_mask(digit_labels, digit_mask)
        digit_outputs_filtered = tf.boolean_mask(digit_outputs, digit_mask)
        y_pred_class = tf.argmax(digit_outputs_filtered, axis=1)
        return tf.reduce_mean(tf.cast(tf.equal(tf.cast(digit_labels_filtered, tf.int64), y_pred_class), tf.float32))
    else:
        return 0.0


def binary_accuracy(y_true, y_pred):
    """
    Binary accuracy for digit detection.
    """
    _, is_digit_labels = y_true
    _, binary_output = y_pred
    
    # Binary prediction: > 0 means "is digit"
    binary_pred = tf.cast(binary_output > 0, tf.float32)
    is_digit_labels = tf.cast(is_digit_labels, tf.float32)
    
    return tf.reduce_mean(tf.cast(tf.equal(binary_pred, is_digit_labels), tf.float32))


def create_cnn_svm_model():
    """
    Create a CNN + SVM hybrid model with two outputs:
    1. 10 outputs for digit classification (multi-class hinge)
    2. 1 output for digit detection (binary hinge)
    
    Architecture:
    - CNN layers for feature extraction
    - Two branches: digit classification + binary detection
    - Uses hybrid loss (multi-class + binary hinge)
    
    Returns:
        Compiled Keras model
    """
    # Model capacity
    number_convolution_channels = 32
    number_convolution_channelsF = 64
    neurons_in_dense_layer = 64
    
    # Shared CNN feature extraction
    input_layer = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(number_convolution_channels, (3, 3), activation='elu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(number_convolution_channels, (3, 3), activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(number_convolution_channelsF, (3, 3), activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(neurons_in_dense_layer, activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Two output branches
    # 1. Digit classification: 10 outputs (0-9) with linear activation
    digit_output = layers.Dense(10, activation='linear', name='digit_output',
                                kernel_regularizer=keras.regularizers.l2(0.001))(x)
    
    # 2. Binary detection: 1 output (is digit?) with linear activation
    binary_output = layers.Dense(1, activation='linear', name='binary_output',
                                 kernel_regularizer=keras.regularizers.l2(0.001))(x)
    
    # Create model with two outputs
    model = keras.Model(inputs=input_layer, outputs=[digit_output, binary_output])
    
    # Compile with separate losses for each output
    model.compile(
        optimizer='adam',
        loss={'digit_output': digit_loss, 'binary_output': binary_loss},
        metrics={'digit_output': svm_accuracy, 'binary_output': binary_accuracy}
    )
    
    return model


def load_or_create_cnn_svm_classifier(classifier_model_path=None, 
                                     train_model=True,
                                     num_epochs=20):
    """
    Load or create and train a CNN+SVM digit classifier.
    
    Args:
        classifier_model_path: Path to save/load model
        train_model: Whether to train the model
        num_epochs: Number of training epochs
    
    Returns:
        Trained Keras model
    """
    if classifier_model_path and os.path.exists(classifier_model_path):
        try:
            model = keras.models.load_model(classifier_model_path, 
                                           custom_objects={'digit_loss': digit_loss,
                                                          'binary_loss': binary_loss,
                                                          'svm_accuracy': svm_accuracy,
                                                          'binary_accuracy': binary_accuracy})
            print("CNN+SVM hybrid classifier loaded successfully")
            return model
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Creating new model...")
    
    # Create model
    model = create_cnn_svm_model()
    print("CNN+SVM hybrid model created (10 digit outputs + 1 binary output)")
    model.summary()
    
    if not train_model:
        return model
    
    # Load datasets
    print("\nLoading datasets...")
    
    # Load EMNIST Digits
    if EMNIST_AVAILABLE:
        try:
            x_train_digits, y_train_digits = extract_training_samples('digits')
            x_test_digits, y_test_digits = extract_test_samples('digits')
            x_train_digits = x_train_digits.astype('float32') / 255.0
            x_test_digits = x_test_digits.astype('float32') / 255.0
            x_train_digits = x_train_digits.reshape(-1, 28, 28, 1)
            x_test_digits = x_test_digits.reshape(-1, 28, 28, 1)
            print(f"Loaded EMNIST: {len(x_train_digits):,} training, {len(x_test_digits):,} test samples")
        except Exception as e:
            print(f"Warning: Could not load EMNIST: {e}")
            # Fallback to MNIST
            (x_train_digits, y_train_digits), (x_test_digits, y_test_digits) = keras.datasets.mnist.load_data()
            x_train_digits = x_train_digits.astype('float32') / 255.0
            x_test_digits = x_test_digits.astype('float32') / 255.0
            x_train_digits = x_train_digits.reshape(-1, 28, 28, 1)
            x_test_digits = x_test_digits.reshape(-1, 28, 28, 1)
            print(f"Loaded MNIST: {len(x_train_digits):,} training, {len(x_test_digits):,} test samples")
    else:
        # Use MNIST
        (x_train_digits, y_train_digits), (x_test_digits, y_test_digits) = keras.datasets.mnist.load_data()
        x_train_digits = x_train_digits.astype('float32') / 255.0
        x_test_digits = x_test_digits.astype('float32') / 255.0
        x_train_digits = x_train_digits.reshape(-1, 28, 28, 1)
        x_test_digits = x_test_digits.reshape(-1, 28, 28, 1)
        print(f"Loaded MNIST: {len(x_train_digits):,} training, {len(x_test_digits):,} test samples")
    
    # Create negative examples (15% ratio)
    print("\nCreating negative examples...")
    x_train_neg, _ = create_negative_examples(len(x_train_digits), target_ratio=0.15)
    x_test_neg, _ = create_negative_examples(len(x_test_digits), target_ratio=0.15)
    print(f"Created {len(x_train_neg):,} training negatives, {len(x_test_neg):,} test negatives")
    
    # Combine digits and negatives
    x_train = np.concatenate([x_train_digits, x_train_neg], axis=0)
    x_test = np.concatenate([x_test_digits, x_test_neg], axis=0)
    
    # Prepare labels for hybrid model:
    # - digit_labels: 0-9 for digits, -1 for negatives
    # - is_digit_labels: 1 for digits, 0 for negatives
    train_digit_labels = y_train_digits.astype(np.int32)
    train_is_digit = np.ones(len(train_digit_labels), dtype=np.int32)
    train_neg_labels = np.full(len(x_train_neg), -1, dtype=np.int32)
    train_neg_is_digit = np.zeros(len(x_train_neg), dtype=np.int32)
    
    y_train_digit_labels = np.concatenate([train_digit_labels, train_neg_labels])
    y_train_is_digit = np.concatenate([train_is_digit, train_neg_is_digit])
    
    test_digit_labels = y_test_digits.astype(np.int32)
    test_is_digit = np.ones(len(test_digit_labels), dtype=np.int32)
    test_neg_labels = np.full(len(x_test_neg), -1, dtype=np.int32)
    test_neg_is_digit = np.zeros(len(x_test_neg), dtype=np.int32)
    
    y_test_digit_labels = np.concatenate([test_digit_labels, test_neg_labels])
    y_test_is_digit = np.concatenate([test_is_digit, test_neg_is_digit])
    
    # Shuffle training data
    train_indices = np.random.permutation(len(x_train))
    x_train = x_train[train_indices]
    y_train_digit_labels = y_train_digit_labels[train_indices]
    y_train_is_digit = y_train_is_digit[train_indices]
    
    # Shuffle test data
    test_indices = np.random.permutation(len(x_test))
    x_test = x_test[test_indices]
    y_test_digit_labels = y_test_digit_labels[test_indices]
    y_test_is_digit = y_test_is_digit[test_indices]
    
    print(f"\nTraining data: {len(x_train_digits):,} digits + {len(x_train_neg):,} negatives = {len(x_train):,} total")
    print(f"Test data: {len(x_test_digits):,} digits + {len(x_test_neg):,} negatives = {len(x_test):,} total")
    
    # Train model
    print(f"\nTraining CNN+SVM hybrid model for {num_epochs} epochs...")
    history = model.fit(
        x_train,
        {'digit_output': y_train_digit_labels, 'binary_output': y_train_is_digit},
        batch_size=64,
        epochs=num_epochs,
        validation_data=(x_test, {'digit_output': y_test_digit_labels, 'binary_output': y_test_is_digit}),
        verbose=1
    )
    
    # Save model
    if classifier_model_path:
        os.makedirs(os.path.dirname(classifier_model_path), exist_ok=True)
        model.save(classifier_model_path)
        print(f"\nModel saved to: {classifier_model_path}")
    
    # Evaluate
    test_results = model.evaluate(
        x_test,
        {'digit_output': y_test_digit_labels, 'binary_output': y_test_is_digit},
        verbose=0
    )
    print(f"\nTest results:")
    print(f"  Loss: {test_results[0]:.4f}")
    print(f"  Digit accuracy: {test_results[1]:.4f}")
    print(f"  Binary accuracy: {test_results[2]:.4f}")
    
    return model


def classify_digit(classifier_model, digit_image):
    """
    Classify a single digit image using the CNN+SVM hybrid model.
    
    Args:
        classifier_model: Trained Keras model (CNN+SVM hybrid)
        digit_image: 28x28 greyscale image (numpy array)
    
    Returns:
        Predicted digit (0-9) and confidence score (raw SVM score)
        Returns (-1, 0.0) if binary output indicates "not a digit"
    """
    # Ensure image is the right shape and type
    if digit_image.shape != (28, 28):
        # Resize if needed
        digit_image = cv2.resize(digit_image, (28, 28))
    
    # Normalize pixel values to [0, 1]
    digit_normalized = digit_image.astype('float32') / 255.0
    
    # The input image should already be in MNIST format: white digits on black background
    # MNIST: white digits (high values ~1.0) on black background (low values ~0.0)
    
    # Reshape for model input: (1, 28, 28, 1)
    digit_input = digit_normalized.reshape(1, 28, 28, 1)
    
    # Predict (model has two outputs: digit_outputs, binary_output)
    digit_outputs, binary_output = classifier_model.predict(digit_input, verbose=0)
    
    # Check binary output first: if < 0, it's not a digit
    is_digit = binary_output[0] > 0
    
    if not is_digit:
        # Not a digit - return -1 with confidence 0
        return -1, 0.0
    
    # It's a digit - use digit outputs
    predicted_digit = np.argmax(digit_outputs[0])
    confidence = float(digit_outputs[0][predicted_digit])  # Raw SVM score
    
    return int(predicted_digit), confidence


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train CNN+SVM digit classifier')
    parser.add_argument('--model-path', type=str, help='Path to save model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--no-train', action='store_true', help='Load model without training')
    args = parser.parse_args()
    
    model = load_or_create_cnn_svm_classifier(
        classifier_model_path=args.model_path,
        train_model=not args.no_train,
        num_epochs=args.epochs
    )
