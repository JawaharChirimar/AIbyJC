#!/usr/bin/env python3
"""
DigitClassifierSoftMax11.py

Provides functions for creating, training, and using a CNN-based digit classifier.
Uses softmax with 11 classes: 10 digits (0-9) + 1 "not a digit" class (10).
"""

import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import augmentation module
from DataManagement.DataAugmentation import (
    ImageDataGeneratorWithAugmentation,
    upscale_to_64x64,
    batch_upscale_to_64x64
)

# Import non-digit generator
from DataManagement.NonDigitGenerator import (
    create_negative_examples,
    load_custom_non_digits,
    NEGATIVE_RATIO,
    DATA_DIR
)
# Import pre-generated non-digits loader
from DataManagement.PregenNonDigits import load_non_digits
# Import pre-generated augmented data loader
from DataManagement.PregenAugmentedData import load_augmented_data

# =============================================================================
# CONFIGURABLE CONSTANTS
# =============================================================================
DROPOUT_RATE = 0.5          # Dropout rate in model (prevents overfitting)
SIGMOID_THRESHOLD = 0.5     # Threshold for sigmoid classification (output > threshold = detected)
LEARNING_RATE = 0.0005       # Learning rate for Adam optimizer (default: 0.001)
BATCH_SIZE = 128             # Batch size for training


def sigmoid_accuracy(y_true, y_pred):
    """
    Custom accuracy metric for sigmoid mode that correctly handles negatives.
    
    For digits: argmax(pred) == argmax(true)
    For negatives: max(pred) < threshold (all outputs should be low)
    """
    # Identify negatives: all zeros in true labels
    is_negative = tf.reduce_sum(y_true, axis=1) < 0.5  # All zeros = negative
    
    # For negatives: check if max prediction < threshold
    max_pred = tf.reduce_max(y_pred, axis=1)
    negative_correct = tf.cast(max_pred < SIGMOID_THRESHOLD, tf.float32)
    
    # For digits: check if argmax matches
    pred_class = tf.argmax(y_pred, axis=1)
    true_class = tf.argmax(y_true, axis=1)
    digit_correct = tf.cast(tf.equal(pred_class, true_class), tf.float32)
    
    # Combine: use negative_correct for negatives, digit_correct for digits
    is_negative_float = tf.cast(is_negative, tf.float32)
    accuracy = is_negative_float * negative_correct + (1.0 - is_negative_float) * digit_correct
    
    return tf.reduce_mean(accuracy)


def create_digit_classifier_model(input_size=28):
    """
    Create a CNN model for digit classification with 11 classes (0-9 digits + 10 "not a digit").
    
    Uses deep model architecture with 4 conv layers, optimized for input_size x input_size images.
    Always uses softmax activation with sparse_categorical_crossentropy loss.
    
    Args:
        input_size: Image size (28 or 64, default: 28)
    
    Returns:
        Compiled Keras model
    """
    # Model capacity for large dataset (240k+ samples)
    number_convolution_channels = 32
    number_convolution_channelsF = 64
    neurons_in_dense_layer = 64  # Increased for 64x64 input
    
    # Always use softmax with 11 classes (0-9 digits + 10 "not a digit")
    output_activation = 'softmax'
    loss_function = 'sparse_categorical_crossentropy'
    output_layer = layers.Dense(11, activation=output_activation)  # 11 classes
    accuracy_metric = 'accuracy'
    
    # Deep model architecture (4 conv layers) for input_size x input_size input
    #conv(32) → BN → conv(32) → BN → pool(2,2) → dropout(0.25) → 
    #conv(64) → BN → conv(64) → BN → pool(2,2) → dropout(0.25) → 
    #flatten → dense(128) → BN → dropout(0.5) → dense(11) 
    model = keras.Sequential([
        layers.Input(shape=(input_size, input_size, 1)),
        layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(number_convolution_channelsF, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.Conv2D(number_convolution_channelsF, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(number_convolution_channelsF * 2, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(neurons_in_dense_layer, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        output_layer  # 11 classes: 0-9 digits + 10 "not a digit"
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_function,
        metrics=[accuracy_metric]
    )
    
    return model


class AugmentationStatsCallback(keras.callbacks.Callback):
    """
    Custom callback to print per-epoch augmentation statistics from DataAugmentation.py.
    """
    def __init__(self, datagen):
        super().__init__()
        self.datagen = datagen
    
    def on_epoch_end(self, epoch, logs=None):
        # Print epoch stats and reset for next epoch
        self.datagen.print_epoch_stats(epoch_num=epoch+1)
        self.datagen.on_epoch_end()  # Update total stats
        self.datagen.reset_epoch_stats()


class Softmax11DiagnosticsCallback(keras.callbacks.Callback):
    """
    Callback to print per-epoch diagnostics for softmax 11-class mode.
    Shows digit accuracy (0-9) vs negative rejection rate (class 10).
    """
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        # Precompute masks (they don't change)
        # Digits are classes 0-9, negatives are class 10
        self.digit_mask = y_val < 10
        self.negative_mask = y_val == 10
        self.n_digits = np.sum(self.digit_mask)
        self.n_negatives = np.sum(self.negative_mask)
        self.digit_labels = y_val[self.digit_mask] if self.n_digits > 0 else None
    
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        y_pred = self.model.predict(self.x_val, verbose=0)
        
        results = []
        
        # Digit classification accuracy (classes 0-9)
        if self.n_digits > 0:
            digit_preds = np.argmax(y_pred[self.digit_mask], axis=1)
            digit_acc = np.mean(digit_preds == self.digit_labels) * 100
            # Check confidence (max output > 0.5)
            digit_max_outputs = np.max(y_pred[self.digit_mask], axis=1)
            digit_confident = np.mean(digit_max_outputs > 0.5) * 100
            results.append(f"Digits: {digit_acc:.1f}% (conf>0.5: {digit_confident:.1f}%)")
        
        # Negative rejection rate (class 10 should be predicted for negatives)
        if self.n_negatives > 0:
            neg_preds = np.argmax(y_pred[self.negative_mask], axis=1)
            neg_rejected = np.sum(neg_preds == 10)
            neg_acc = neg_rejected / self.n_negatives * 100
            results.append(f"Neg rejected: {neg_acc:.1f}% ({neg_rejected}/{self.n_negatives})")
            
            # Show distribution stats for negatives
            neg_preds_probs = y_pred[self.negative_mask]
            neg_class10_probs = neg_preds_probs[:, 10]  # Probability assigned to class 10
            neg_mean = np.mean(neg_class10_probs)
            neg_median = np.median(neg_class10_probs)
            neg_max_val = np.max(neg_class10_probs)
            neg_std = np.std(neg_class10_probs)
            
            # Show stats every epoch to monitor
            if epoch == 0 or (epoch + 1) % 5 == 0:  # Show detailed stats on first epoch and every 5 epochs
                print(f"    Neg stats (class 10 prob): mean={neg_mean:.3f}, median={neg_median:.3f}, max={neg_max_val:.3f}, std={neg_std:.3f}")
        
        print(f"  [Softmax11] {' | '.join(results)}")


class SigmoidDiagnosticsCallback(keras.callbacks.Callback):
    """
    Callback to print per-epoch diagnostics for sigmoid mode.
    Shows digit accuracy vs negative rejection rate after each epoch.
    """
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        # Precompute masks (they don't change)
        self.digit_mask = np.sum(y_val, axis=1) > 0
        self.negative_mask = ~self.digit_mask
        self.n_digits = np.sum(self.digit_mask)
        self.n_negatives = np.sum(self.negative_mask)
        self.digit_labels = np.argmax(y_val[self.digit_mask], axis=1) if self.n_digits > 0 else None
    
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        y_pred = self.model.predict(self.x_val, verbose=0)
        
        results = []
        
        # Digit classification accuracy
        if self.n_digits > 0:
            digit_preds = np.argmax(y_pred[self.digit_mask], axis=1)
            digit_acc = np.mean(digit_preds == self.digit_labels) * 100
            # Check confidence (max output > 0.5)
            digit_max_outputs = np.max(y_pred[self.digit_mask], axis=1)
            digit_confident = np.mean(digit_max_outputs > 0.5) * 100
            results.append(f"Digits: {digit_acc:.1f}% (conf>0.5: {digit_confident:.1f}%)")
        
        # Negative rejection rate
        if self.n_negatives > 0:
            neg_max = np.max(y_pred[self.negative_mask], axis=1)
            neg_rejected = np.sum(neg_max < SIGMOID_THRESHOLD)
            neg_acc = neg_rejected / self.n_negatives * 100
            # Show distribution stats to detect if negatives are too easy
            neg_mean = np.mean(neg_max)
            neg_std = np.std(neg_max)
            neg_max_val = np.max(neg_max)
            neg_median = np.median(neg_max)
            results.append(f"Neg rejected: {neg_acc:.1f}% ({neg_rejected}/{self.n_negatives})")
            # Show stats every epoch to monitor
            if epoch == 0 or (epoch + 1) % 5 == 0:  # Show detailed stats on first epoch and every 5 epochs
                print(f"    Neg stats: mean={neg_mean:.3f}, median={neg_median:.3f}, max={neg_max_val:.3f}, std={neg_std:.3f}")
            # Add warning if negatives seem too easy
            if neg_mean < 0.1 and neg_max_val < 0.3:
                print(f"    WARNING: Negatives may be too easy! Consider making them more challenging.")
        
        print(f"  [Sigmoid] {' | '.join(results)}")


def load_font_digits(split='train', input_size=28):
    """
    Load font-generated digit images (already augmented).
    
    Args:
        split: 'train' or 'test'
        input_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (x_data, y_data) or (None, None) if not available
    """
    # Try size-specific filename first, then fallback to old format
    npz_path = DATA_DIR / "font_digits" / f"font_digits_{split}_{input_size}x{input_size}_softmax.npz"
    if not npz_path.exists():
        # Fallback to old format (for backward compatibility)
        npz_path = DATA_DIR / "font_digits" / f"font_digits_{split}_softmax.npz"
    
    if npz_path.exists():
        try:
            data = np.load(npz_path)
            x_data = data['x'].astype(np.float32)
            # FontDigitGenerator saves labels as 'y', not 'y_softmax'
            y_data = data['y'].astype(np.int32)
            # Ensure normalized and correct shape
            if x_data.max() > 1.0:
                x_data = x_data / 255.0
            if len(x_data.shape) == 3:
                # Infer size from data shape (should be square)
                size = x_data.shape[1]  # Assuming square images
                x_data = x_data.reshape(-1, size, size, 1)
            print(f"  Loaded font digits ({split}): {len(x_data):,} samples")
            return x_data, y_data
        except Exception as e:
            print(f"Warning: Could not load font digits from {npz_path}: {e}")
            return None, None
    else:
        print(f"Font digits not found at {npz_path}")
        print("Generate with: python DataManagement/FontDigitGenerator.py --api-key YOUR_KEY")
        return None, None


def load_custom_one(split='train'):
    """
    Load custom "1" digit variations (serif style with no base).
    
    Args:
        split: 'train' or 'test'
    
    Returns:
        Tuple of (x_data, y_data) or (None, None) if not available
    """
    npz_path = DATA_DIR / "custom_one" / f"custom_one_{split}_softmax.npz"
    
    if npz_path.exists():
        try:
            data = np.load(npz_path)
            x_data = data['x'].astype(np.float32)
            # GenerateCustomOne saves labels as 'y', not 'y_softmax'
            y_data = data['y'].astype(np.int32)
            # Ensure normalized and correct shape
            if x_data.max() > 1.0:
                x_data = x_data / 255.0
            if len(x_data.shape) == 3:
                # Infer size from data shape (should be square)
                size = x_data.shape[1]  # Assuming square images
                x_data = x_data.reshape(-1, size, size, 1)
            print(f"  Loaded custom '1' ({split}): {len(x_data):,} samples")
            return x_data, y_data
        except Exception as e:
            print(f"Warning: Could not load custom '1' from {npz_path}: {e}")
            return None, None
    else:
        print(f"Custom '1' digits not found at {npz_path}")
        print("Generate with: python DataManagement/GenerateCustomOne.py")
        return None, None


def load_and_combine_datasets(test_only=False, input_size=28):
    """
    Load and combine all digit datasets (all pre-generated at input_size x input_size):
    - EMNIST Digits (pre-generated)
    - ARDIS (pre-generated)
    - USPS (pre-generated)
    - Font Digits (already augmented - skip augmentation)
    - Custom "1" digits (pre-generated)
    
    Args:
        test_only: If True, only load test data (for FAST mode with pre-gen augmented data)
        input_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test, is_google_fonts_train)
        - x_train/y_train: All training data combined (None if test_only=True)
        - x_test/y_test: All test data combined
        - is_google_fonts_train: Boolean array marking Google Font samples (None if test_only=True)
        Arrays are normalized to [0, 1] and shaped as (samples, input_size, input_size, 1)
    """
    train_datasets = []
    test_datasets = []
    train_names = []
    is_google_fonts_arrays = []  # Track which samples are from Google Fonts
    
    if test_only:
        print("Loading TEST data only (training data from pre-generated augmented file)...")
    else:
        print(f"Loading pre-generated {input_size}x{input_size} datasets...")
    
    # =========================================================================
    # EMNIST Digits
    # =========================================================================
    emnist_train_path = DATA_DIR / "EMNIST" / f"emnist_digits_train_{input_size}x{input_size}.npz"
    emnist_test_path = DATA_DIR / "EMNIST" / f"emnist_digits_test_{input_size}x{input_size}.npz"
    
    if emnist_test_path.exists() and (test_only or emnist_train_path.exists()):
        try:
            test_data = np.load(emnist_test_path)
            x_test = test_data['x'].astype(np.float32)
            y_test = test_data['y_softmax'].astype(np.int32)
            
            if x_test.max() > 1.0:
                x_test = x_test / 255.0
            if len(x_test.shape) == 3:
                x_test = x_test.reshape(-1, input_size, input_size, 1)
            test_datasets.append((x_test, y_test))
            
            if not test_only:
                train_data = np.load(emnist_train_path)
                x_train = train_data['x'].astype(np.float32)
                y_train = train_data['y_softmax'].astype(np.int32)
                if x_train.max() > 1.0:
                    x_train = x_train / 255.0
                if len(x_train.shape) == 3:
                    x_train = x_train.reshape(-1, input_size, input_size, 1)
                train_datasets.append((x_train, y_train))
                train_names.append(f"EMNIST ({len(x_train):,})")
                is_google_fonts_arrays.append(np.zeros(len(x_train), dtype=bool))
                print(f"  EMNIST: {len(x_train):,} train, {len(x_test):,} test")
            else:
                print(f"  EMNIST: {len(x_test):,} test")
        except Exception as e:
            print(f"  Warning: Could not load EMNIST {input_size}x{input_size}: {e}")
    else:
        print(f"  Warning: EMNIST {input_size}x{input_size} not found. Run PregenData{input_size}.py first.")
    
    # =========================================================================
    # ARDIS
    # =========================================================================
    ardis_train_path = DATA_DIR / "ardis" / f"ardis_train_{input_size}x{input_size}.npz"
    ardis_test_path = DATA_DIR / "ardis" / f"ardis_test_{input_size}x{input_size}.npz"
    
    if ardis_test_path.exists() and (test_only or ardis_train_path.exists()):
        try:
            test_data = np.load(ardis_test_path)
            x_test = test_data['x'].astype(np.float32)
            y_test = test_data['y_softmax'].astype(np.int32)
            if x_test.max() > 1.0:
                x_test = x_test / 255.0
            if len(x_test.shape) == 3:
                x_test = x_test.reshape(-1, input_size, input_size, 1)
            test_datasets.append((x_test, y_test))
            
            if not test_only:
                train_data = np.load(ardis_train_path)
                x_train = train_data['x'].astype(np.float32)
                y_train = train_data['y_softmax'].astype(np.int32)
                if x_train.max() > 1.0:
                    x_train = x_train / 255.0
                if len(x_train.shape) == 3:
                    x_train = x_train.reshape(-1, input_size, input_size, 1)
                train_datasets.append((x_train, y_train))
                train_names.append(f"ARDIS ({len(x_train):,})")
                is_google_fonts_arrays.append(np.zeros(len(x_train), dtype=bool))
                print(f"  ARDIS: {len(x_train):,} train, {len(x_test):,} test")
            else:
                print(f"  ARDIS: {len(x_test):,} test")
        except Exception as e:
            print(f"  Warning: Could not load ARDIS {input_size}x{input_size}: {e}")
    else:
        print(f"  Warning: ARDIS {input_size}x{input_size} not found. Run PregenData{input_size}.py first.")
    
    # =========================================================================
    # USPS
    # =========================================================================
    usps_train_path = DATA_DIR / "usps" / f"usps_train_{input_size}x{input_size}.npz"
    usps_test_path = DATA_DIR / "usps" / f"usps_test_{input_size}x{input_size}.npz"
    
    if usps_test_path.exists() and (test_only or usps_train_path.exists()):
        try:
            test_data = np.load(usps_test_path)
            x_test = test_data['x'].astype(np.float32)
            y_test = test_data['y_softmax'].astype(np.int32)
            if x_test.max() > 1.0:
                x_test = x_test / 255.0
            if len(x_test.shape) == 3:
                x_test = x_test.reshape(-1, input_size, input_size, 1)
            test_datasets.append((x_test, y_test))
            
            if not test_only:
                train_data = np.load(usps_train_path)
                x_train = train_data['x'].astype(np.float32)
                y_train = train_data['y_softmax'].astype(np.int32)
                if x_train.max() > 1.0:
                    x_train = x_train / 255.0
                if len(x_train.shape) == 3:
                    x_train = x_train.reshape(-1, input_size, input_size, 1)
                train_datasets.append((x_train, y_train))
                train_names.append(f"USPS ({len(x_train):,})")
                is_google_fonts_arrays.append(np.zeros(len(x_train), dtype=bool))
                print(f"  USPS: {len(x_train):,} train, {len(x_test):,} test")
            else:
                print(f"  USPS: {len(x_test):,} test")
        except Exception as e:
            print(f"  Warning: Could not load USPS {input_size}x{input_size}: {e}")
    else:
        print(f"  Warning: USPS {input_size}x{input_size} not found. Run PregenData{input_size}.py first.")
    
    # =========================================================================
    # Font Digits (already augmented - SKIP augmentation)
    # =========================================================================
    if not test_only:
        print("\nLoading font-generated digits (already augmented)...")
        x_fonts_train, y_fonts_train = load_font_digits(split='train')
        if x_fonts_train is not None:
            train_datasets.append((x_fonts_train, y_fonts_train))
            train_names.append(f"Fonts ({len(x_fonts_train):,})")
            is_google_fonts_arrays.append(np.ones(len(x_fonts_train), dtype=bool))
    
    x_fonts_test, y_fonts_test = load_font_digits(split='test')
    if x_fonts_test is not None:
        test_datasets.append((x_fonts_test, y_fonts_test))
        if test_only:
            print(f"  Font digits: {len(x_fonts_test):,} test")
    
    # =========================================================================
    # Custom "1" digits
    # =========================================================================
    if not test_only:
        print("\nLoading custom '1' digits...")
        x_custom_train, y_custom_train = load_custom_one(split='train')
        if x_custom_train is not None:
            train_datasets.append((x_custom_train, y_custom_train))
            train_names.append(f"CustomOne ({len(x_custom_train):,})")
            is_google_fonts_arrays.append(np.zeros(len(x_custom_train), dtype=bool))
    
    x_custom_test, y_custom_test = load_custom_one(split='test')
    if x_custom_test is not None:
        test_datasets.append((x_custom_test, y_custom_test))
        if test_only:
            print(f"  Custom '1': {len(x_custom_test):,} test")
    
    # =========================================================================
    # COMBINE DATASETS
    # =========================================================================
    
    if len(test_datasets) == 0:
        raise ValueError(f"No test datasets found! Run PregenData{input_size}.py to generate {input_size}x{input_size} datasets.")
    
    # Combine test data (always needed)
    x_test = np.concatenate([ds[0] for ds in test_datasets], axis=0)
    y_test = np.concatenate([ds[1] for ds in test_datasets], axis=0)
    if len(x_test.shape) == 3:
        x_test = x_test.reshape(-1, input_size, input_size, 1)
    
    if test_only:
        # Return only test data
        print(f"\n  Total test: {len(x_test):,} samples")
        return None, None, x_test, y_test, None
    
    # Combine training data (only when not test_only)
    if len(train_datasets) == 0:
        raise ValueError(f"No training datasets found! Run PregenData{input_size}.py to generate {input_size}x{input_size} datasets.")
    
    x_train = np.concatenate([ds[0] for ds in train_datasets], axis=0)
    y_train = np.concatenate([ds[1] for ds in train_datasets], axis=0)
    is_google_fonts_train = np.concatenate(is_google_fonts_arrays, axis=0)
    
    if len(x_train.shape) == 3:
        x_train = x_train.reshape(-1, input_size, input_size, 1)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset Summary (all {input_size}x{input_size})")
    print(f"{'='*60}")
    print(f"Training datasets: {' + '.join(train_names)}")
    print(f"  Total training: {len(x_train):,} samples")
    print(f"  Google Fonts (skip augmentation): {np.sum(is_google_fonts_train):,}")
    print(f"  Other (will be augmented): {np.sum(~is_google_fonts_train):,}")
    print(f"Test set: {len(x_test):,} samples")
    print(f"{'='*60}\n")
    
    return x_train, y_train, x_test, y_test, is_google_fonts_train


def load_or_create_digit_classifier(classifier_model_path=None, 
train_model=True, num_epochs=20, use_stratified=False, input_size=28, initial_model_path=None):
    """
    Load a pre-trained digit classifier or create/train a new one.
    
    Always uses EMNIST (if available), data augmentation, and 4 layer CNN model.
    
    Args:
        classifier_model_path: Path to save the trained model (.keras file)
        train_model: Whether to train a new model (True) or load existing (False)
        num_epochs: Number of training epochs (default: 20)
        use_stratified: Use stratified batch sampling (default: False)
        input_size: Image size (28 or 64, default: 28)
        initial_model_path: Path to pre-trained model to use as starting point (optional)
        Always uses softmax with 11 classes (0-9 digits + 10 "not a digit")
    
    Returns:
        Trained Keras model for digit classification
    """

    print("===========train_model: ", train_model)
    print("===========classifier_model_path: ", classifier_model_path)
    
    # CRITICAL: If train_model is False, we MUST load an existing model - do NOT train
    # Return early - never reach training code below
    if not train_model:
        if classifier_model_path is None or classifier_model_path == '' or not classifier_model_path:
            raise ValueError("classifier_model_path must be provided when train_model=False")
        
        classifier_model_path = str(classifier_model_path)  # Convert Path to string if needed
        if not os.path.exists(classifier_model_path):
            raise ValueError(f"Model file not found: {classifier_model_path}. Cannot load model when train_model=False.")
        
        try:
            print(f"Loading digit classifier from: {classifier_model_path}")
            # No custom objects needed for softmax model
            model = keras.models.load_model(classifier_model_path)
            print("Digit classifier loaded successfully - RETURNING, will NOT train")
            return model  # RETURN HERE - DO NOT CONTINUE TO TRAINING CODE BELOW
        except Exception as e:
            raise ValueError(f"Cannot load model from {classifier_model_path}: {e}. Set train_model=True to create a new model.")
    
    # Only reach here if train_model=True - we're going to train a new model
    print("DEBUG: train_model is True, proceeding to training")
    # Create the run directory now
    # Create timestamped directory for model checkpoints
    base_dir = DATA_DIR / "modelForDE"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory: run_YYYY_MM_DD_HH_MM_SS
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model checkpoints will be saved to: {run_dir}")
    
    # Load initial model if provided, otherwise create new model
    if initial_model_path is not None and os.path.exists(initial_model_path):
        print(f"Loading initial model from: {initial_model_path}")
        print(f"  Will continue training from this checkpoint...")
        try:
            model = keras.models.load_model(initial_model_path)
            print(f"  Initial model loaded successfully")
        except Exception as e:
            print(f"  Warning: Could not load initial model: {e}")
            print(f"  Creating new model instead...")
            model = create_digit_classifier_model(input_size=input_size)
    else:
        # Create new model (always uses softmax with 11 classes)
        if initial_model_path is not None:
            print(f"Warning: Initial model path provided but file not found: {initial_model_path}")
            print(f"  Creating new model instead...")
        else:
            print(f"Creating new digit classifier model ({input_size}x{input_size} input, 11 classes)...")
        model = create_digit_classifier_model(input_size=input_size)
    
    # Try to train on all digit datasets
    try:
        # =====================================================================
        # Try to load PRE-GENERATED augmented data first (FAST training!)
        # =====================================================================
        x_train_aug, y_train_aug = load_augmented_data(image_size=input_size)
        use_pregen = x_train_aug is not None
        
        if use_pregen:
            print("\n✓ Using PRE-GENERATED augmented data (fast training mode)")
            x_train_all = x_train_aug
            y_train_all = y_train_aug
            
            # Only load test data (training already in pre-gen file)
            _, _, x_test, y_test, _ = load_and_combine_datasets(test_only=True, input_size=input_size)
            _, _, x_negative_test, y_negative_test = load_non_digits(image_size=input_size)
            
            if x_negative_test is not None:
                x_test = np.concatenate([x_test, x_negative_test], axis=0)
                y_test = np.concatenate([y_test, y_negative_test], axis=0)
                indices = np.random.permutation(len(x_test))
                x_test = x_test[indices]
                y_test = y_test[indices]
            
            print(f"Training samples: {len(x_train_all):,} (pre-augmented)")
            print(f"Test samples: {len(x_test):,}")
            print(f"Number of epochs: {num_epochs}")
        else:
            # =====================================================================
            # Fall back to ON-THE-FLY augmentation (slow but works)
            # =====================================================================
            print("\n⚠ Pre-generated augmented data not found. Using on-the-fly augmentation (slower)...")
            print("  Run: python3 PregenAugmentedData.py to speed up future training.\n")
            
            # Load and combine all datasets
            x_train, y_train, x_test, y_test, is_google_fonts_train = load_and_combine_datasets(input_size=input_size)
            
            # Store for digit tracking
            x_digits_test = x_test.copy()
            y_digits_test = y_test.copy()
            total_digits = len(x_train)
            
            # Load pre-generated negative examples (non-digits) labeled as class 10
            print("\nLoading pre-generated non-digits for softmax training (11 classes)...")
            x_negative_train, y_negative_train, x_negative_test, y_negative_test = load_non_digits(image_size=input_size)
            
            if x_negative_train is None:
                # Fall back to on-the-fly generation if pre-generated not available
                print("  Pre-generated not found. Generating on-the-fly (slower, uses more RAM)...")
                print(f"  Target ratio: {NEGATIVE_RATIO*100:.0f}% of digit samples")
                x_negative_train, y_negative_train = create_negative_examples(total_digits, target_ratio=NEGATIVE_RATIO)
                x_negative_test, y_negative_test = create_negative_examples(len(x_digits_test), target_ratio=NEGATIVE_RATIO)
            
            print(f"  Total negative examples - Train: {len(x_negative_train):,}, Test: {len(x_negative_test):,}")
            
            # Print per-class distribution for digits (0-9)
            print(f"\n=== Per-Class Distribution (Training) ===")
            unique_digits, digit_counts = np.unique(y_train, return_counts=True)
            for digit, count in zip(unique_digits, digit_counts):
                if digit < 10:  # Only digits 0-9
                    print(f"  Digit {digit}: {count:,} samples ({count/total_digits*100:.1f}% of digits)")
            print(f"  Non-digits (class 10): {len(x_negative_train):,} samples")
            print(f"==========================================\n")
            
            # Combine negatives with digits for training (negatives labeled as 10)
            x_train_all = np.concatenate([x_train, x_negative_train], axis=0)
            y_train_all = np.concatenate([y_train, y_negative_train], axis=0)
            # Extend is_google_fonts array for negatives (they're NOT Google Fonts)
            is_google_fonts_all = np.concatenate([
                is_google_fonts_train, 
                np.zeros(len(x_negative_train), dtype=bool)
            ], axis=0)
            
            # For test data
            x_test = np.concatenate([x_digits_test, x_negative_test], axis=0)
            y_test = np.concatenate([y_digits_test, y_negative_test], axis=0)
            indices = np.random.permutation(len(x_test))
            x_test = x_test[indices]
            y_test = y_test[indices]
            
            print(f"Training samples: {len(x_train_all):,} ({total_digits:,} digits + {len(x_negative_train):,} non-digits)")
            print(f"Test samples: {len(x_test):,}")
            print(f"Number of epochs: {num_epochs}")
        
        # =====================================================================
        # TRAINING SETUP
        # =====================================================================
        import time
        
        batch_size = BATCH_SIZE
        print(f"Epoch models will be saved as: {run_dir}/digit_classifier_epoch_XX.keras")
        
        # Early stopping settings
        patience = 5  # Stop if no improvement for 5 epochs
        min_delta = 0.0001  # Minimum change to qualify as improvement
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_epoch = 0
        
        if use_pregen:
            # =================================================================
            # FAST MODE: from_generator() with model.fit()
            # =================================================================
            print("\n" + "="*60)
            print("FAST TRAINING MODE (pre-generated augmented data)")
            print("="*60)
            
            steps_per_epoch = len(x_train_all) // batch_size
            val_steps = (len(x_test) + batch_size - 1) // batch_size
            print(f"Training samples: {len(x_train_all):,}")
            print(f"Batch size: {batch_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            
            # Choose generator based on stratified flag
            if use_stratified:
                print("\nUsing STRATIFIED batch sampling (balanced classes per batch)")
                from StratifiedBatchGenerator import create_stratified_batch_generator
                # Create the generator function (returns a callable that returns a generator)
                train_generator = create_stratified_batch_generator(
                    x_train_all, y_train_all, batch_size=batch_size, num_classes=11, non_digit_class=10
                )
            else:
                # Standard random shuffle generator
                def train_generator():
                    n_samples = len(x_train_all)
                    indices = np.arange(n_samples)
                    while True:
                        np.random.shuffle(indices)
                        for start in range(0, n_samples, batch_size):
                            end = min(start + batch_size, n_samples)
                            batch_indices = indices[start:end]
                            yield x_train_all[batch_indices], y_train_all[batch_indices]
            
            # Create tf.data.Dataset from generator (NO memory copy!)
            train_dataset = tf.data.Dataset.from_generator(
                train_generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, input_size, input_size, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.int32)
                )
            )
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
            
            # Validation dataset - use from_tensor_slices for validation (smaller, one-time use)
            # Validation is small enough (~67K samples = ~1GB) that copying is acceptable
            val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            val_dataset = val_dataset.batch(batch_size)
            
            # Custom callback for digit vs non-digit accuracy
            class DigitNonDigitCallback(keras.callbacks.Callback):
                def __init__(self, x_val, y_val):
                    super().__init__()
                    self.x_val = x_val
                    self.y_val = y_val
                
                def on_epoch_end(self, epoch, logs=None):
                    y_pred = np.argmax(self.model.predict(self.x_val, verbose=0), axis=1)
                    digit_mask = self.y_val < 10
                    nondigit_mask = self.y_val == 10
                    digit_acc = np.mean(y_pred[digit_mask] == self.y_val[digit_mask]) if np.sum(digit_mask) > 0 else 0
                    nondigit_acc = np.mean(y_pred[nondigit_mask] == self.y_val[nondigit_mask]) if np.sum(nondigit_mask) > 0 else 0
                    print(f"    Digits: {digit_acc:.4f} ({np.sum(digit_mask):,}) | Non-digits: {nondigit_acc:.4f} ({np.sum(nondigit_mask):,})")
            
            # Callbacks
            callbacks_list = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    min_delta=min_delta,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=str(run_dir / "digit_classifier_best.keras"),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=str(run_dir / "digit_classifier_epoch_{epoch:02d}.keras"),
                    save_freq='epoch',
                    verbose=0
                ),
                DigitNonDigitCallback(x_test, y_test)
            ]
            
            # Train with model.fit()
            history = model.fit(
                train_dataset,
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_dataset,
                validation_steps=val_steps,
                callbacks=callbacks_list,
                verbose=1
            )
            
        else:
            # =================================================================
            # SLOW MODE: On-the-fly augmentation with custom generator
            # =================================================================
            # Create data augmentation generator (from DataAugmentation.py)
            print("\nSetting up data augmentation (from DataAugmentation.py)...")
            train_datagen = ImageDataGeneratorWithAugmentation(
                x_train_all, 
                y_train_all,
                is_google_fonts=is_google_fonts_all,
                augment_ratio=0.20,  # 20% augmented
                num_classes=11,
                non_digit_class=10
            )
            
            # Train the model with augmented data
            print("\n" + "="*60)
            print("Augmentation Configuration (DataAugmentation.py):")
            print("="*60)
            print("  - 20% of each class selected for augmentation")
            print("  - Each augmented sample → 6 images (original + 5 transforms)")
            print("  - Transforms: rotation, shear+, shear-, aspect wide, aspect narrow")
            print("  - Post-processing: 20% blur, 10% thin, 10% thick, 10% erasure, 10% breaks")
            print("  - Google Fonts (already augmented): SKIP augmentation")
            print("  - Blank non-digits: SKIP augmentation")
            print("="*60 + "\n")
            
            # Create callbacks
            stats_callback = AugmentationStatsCallback(train_datagen)
            softmax_callback = Softmax11DiagnosticsCallback(x_test, y_test)
            callbacks_list = [stats_callback, softmax_callback]
            
            # Calculate steps per epoch
            n_google_fonts = np.sum(is_google_fonts_all)
            n_other = len(x_train_all) - n_google_fonts
            expected_images_per_epoch = int(n_google_fonts + n_other * 2.0)
            steps_per_epoch = expected_images_per_epoch // batch_size
            
            print(f"Google Fonts (skip augmentation): {n_google_fonts:,}")
            print(f"Other data (will be augmented 2×): {n_other:,}")
            print(f"Expected images per epoch: ~{expected_images_per_epoch:,}")
            print(f"Steps per epoch: {steps_per_epoch}")
            
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                
                # Reset epoch stats
                train_datagen.reset_epoch_stats()
                
                # Generate augmented data for this epoch
                batch_count = 0
                epoch_loss = 0.0
                epoch_acc = 0.0
                epoch_start = time.time()
                batch_start = time.time()
                
                for batch_x, batch_y in train_datagen.flow(batch_size=batch_size):
                    # Train on batch
                    loss, acc = model.train_on_batch(batch_x, batch_y)
                    epoch_loss += loss
                    epoch_acc += acc
                    batch_count += 1
                    
                    # Progress every 100 batches
                    if batch_count % 100 == 0:
                        avg_loss = epoch_loss / batch_count
                        avg_acc = epoch_acc / batch_count
                        batch_time = time.time() - batch_start
                        print(f"  {batch_count}/{steps_per_epoch} - loss: {avg_loss:.4f} - accuracy: {avg_acc:.4f} - {batch_time:.1f}s/100 batches")
                        batch_start = time.time()
                    
                    if batch_count >= steps_per_epoch:
                        break
                
                # Average metrics for epoch
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                avg_acc = epoch_acc / batch_count if batch_count > 0 else 0
                epoch_time = time.time() - epoch_start
                
                # Evaluate on validation set
                val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
                
                # Separate accuracy for digits vs non-digits
                y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
                digit_mask = y_test < 10
                nondigit_mask = y_test == 10
                
                digit_acc = np.mean(y_pred[digit_mask] == y_test[digit_mask]) if np.sum(digit_mask) > 0 else 0
                nondigit_acc = np.mean(y_pred[nondigit_mask] == y_test[nondigit_mask]) if np.sum(nondigit_mask) > 0 else 0
                
                print(f"  {batch_count}/{steps_per_epoch} - loss: {avg_loss:.4f} - accuracy: {avg_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - {epoch_time:.0f}s total")
                print(f"    Digits: {digit_acc:.4f} ({np.sum(digit_mask):,}) | Non-digits: {nondigit_acc:.4f} ({np.sum(nondigit_mask):,})")
                
                # Run callbacks
                logs = {'loss': loss, 'accuracy': acc, 'val_loss': val_loss, 'val_accuracy': val_acc}
                for callback in callbacks_list:
                    callback.model = model
                    callback.on_epoch_end(epoch, logs)
                
                # Save checkpoint
                checkpoint_path = run_dir / f"digit_classifier_epoch_{epoch+1:02d}.keras"
                model.save(str(checkpoint_path))
                
                # Early stopping check
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    # Save best model
                    best_model_path = run_dir / "digit_classifier_best.keras"
                    model.save(str(best_model_path))
                    print(f"    ✓ New best model saved (val_loss: {val_loss:.4f})")
                else:
                    epochs_without_improvement += 1
                    print(f"    No improvement for {epochs_without_improvement}/{patience} epochs (best: epoch {best_epoch}, val_loss: {best_val_loss:.4f})")
                    
                    if epochs_without_improvement >= patience:
                        print(f"\n⚠️  Early stopping triggered! No improvement for {patience} epochs.")
                        print(f"   Best model was at epoch {best_epoch} with val_loss: {best_val_loss:.4f}")
                        break
            
            # Print final augmentation statistics (only in slow mode)
            train_datagen.print_final_stats()
        
        # Save the final model (also saved by checkpoint, but this ensures final state is saved)
        # Save to run_dir
        final_model_path = str(run_dir / "digit_classifier_final.keras")
        
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        print(f"(Individual epoch models saved in: {run_dir})")
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("Evaluating model on test set...")
        print("="*60)
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        num_test_samples = len(x_test)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4%} ({test_accuracy*num_test_samples:.0f} out of {num_test_samples} test images)")
        
        # Get predictions
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Per-class accuracy (softmax mode with 11 classes)
        print("\nPer-class accuracy on test set:")
        print("-" * 40)
        for digit in range(10):
            mask = y_test == digit
            if np.sum(mask) > 0:
                class_accuracy = np.mean(y_pred_classes[mask] == digit)
                correct = np.sum(y_pred_classes[mask] == digit)
                total = np.sum(mask)
                print(f"  Digit {digit}: {class_accuracy:.2%} ({correct}/{total})")
        
        # Non-digit (class 10) accuracy
        mask = y_test == 10
        if np.sum(mask) > 0:
            class_accuracy = np.mean(y_pred_classes[mask] == 10)
            correct = np.sum(y_pred_classes[mask] == 10)
            total = np.sum(mask)
            print(f"  Non-digit: {class_accuracy:.2%} ({correct}/{total})")
        
        print("="*60)
        print("Digit classifier trained and ready!")
        return model
        
    except Exception as e:
        print(f"Warning: Could not train digit classifier: {e}")
        print("Using untrained model (predictions will be random)")
        return model


def classify_digit(classifier_model, digit_image, input_size=28):
    """
    Classify a single digit image using the CNN model with 11 classes.
    
    Args:
        classifier_model: Trained Keras model (11 classes: 0-9 digits + 10 "not a digit")
        digit_image: Greyscale image (numpy array), will be resized to input_size x input_size if needed
        input_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (predicted_digit, confidence)
        - predicted_digit: int (0-9) or -1 if class 10 ("not a digit") is predicted
        - confidence: float (0.0-1.0) or 0.0 if rejected
    """
    # Ensure image is the right shape and type
    if digit_image.shape != (input_size, input_size):
        # Resize using LANCZOS for quality
        digit_image = cv2.resize(digit_image, (input_size, input_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize pixel values to [0, 1]
    digit_normalized = digit_image.astype('float32') / 255.0
    
    # The input image should already be in MNIST format: white digits on black background
    # (ensured by BoundingBoxFromYolo.py preprocessing)
    
    # Reshape for model input: (1, input_size, input_size, 1)
    digit_input = digit_normalized.reshape(1, input_size, input_size, 1)
    
    # Predict
    predictions = classifier_model.predict(digit_input, verbose=0)
    
    # Get predicted class (0-10)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    
    # If class 10 ("not a digit") is predicted, return -1 with the actual confidence
    # Frontend will display the image with -1 to show what was rejected
    if predicted_class == 10:
        return 10, confidence  # Return actual class 10 probability
    
    # Otherwise return the digit (0-9)
    return predicted_class, confidence


def main():
    """
    Standalone training function for the digit classifier.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train a digit classifier on MNIST dataset"
    )
    parser.add_argument(
        "-m", "--model-path",
        type=str,
        default=None,
        help="Path to save the trained model (.keras file). Default: ~/.digit_classifier_mnist.keras"
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="True means train model, False means load model"
    )
    parser.add_argument(
        "--epoch-count",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use stratified batch sampling (balanced classes per batch)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=28,
        choices=[28, 64],
        help="Input image size (28 or 64, default: 28)"
    )
    parser.add_argument(
        "--initial-model",
        type=str,
        default=None,
        help="Path to pre-trained model to use as starting point for training"
    )
    
    args = parser.parse_args()
    
    # Determine input size
    input_size = args.size
        
    # Train the model (always uses softmax with 11 classes)
    print(f"Starting digit classifier training with 11 classes (0-9 digits + 10 'not a digit')...")
    print(f"Input image size: {input_size}x{input_size}")
    model = load_or_create_digit_classifier(
        classifier_model_path=args.model_path, 
        train_model=args.train_model,
        use_stratified=args.stratified,
        num_epochs=args.epoch_count,
        input_size=input_size,
        initial_model_path=args.initial_model
    )
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
