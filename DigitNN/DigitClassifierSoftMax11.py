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

# Import DATA_DIR from NonDigitGenerator (used for data paths)
from DataManagement.NonDigitGenerator import DATA_DIR
# Import pre-generated non-digits loader
from DataManagement.PregenNonDigits import load_non_digits
# Import pre-generated augmented data loader
from DataManagement.PregenAugmentedData import load_augmented_data

# =============================================================================
# CONFIGURABLE CONSTANTS
# =============================================================================
DROPOUT_RATE = 0.5          # Dropout rate in model (prevents overfitting)
LEARNING_RATE = 0.001       # Learning rate for Adam optimizer (default: 0.001)
BATCH_SIZE = 128             # Batch size for training


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


def load_dataset_from_npz(file_path, input_size, label_key='y_softmax', dataset_name=None):
    """
    Helper function to load a dataset from an npz file.
    
    Args:
        file_path: Path to the .npz file
        input_size: Expected image size (28 or 64)
        label_key: Key for labels in npz file ('y_softmax' or 'y', default: 'y_softmax')
        dataset_name: Name of dataset for error messages (optional)
    
    Returns:
        Tuple of (x_data, y_data)
        - x_data: (N, input_size, input_size, 1) float32 [0, 1]
        - y_data: (N,) int32
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data cannot be loaded or is invalid
    """
    if not file_path.exists():
        name = dataset_name or file_path.name
        raise FileNotFoundError(f"Dataset file not found: {file_path} (dataset: {name})")
    
    try:
        data = np.load(file_path)
        x_data = data['x'].astype(np.float32)
        y_data = data[label_key].astype(np.int32)
        
        # Check if data is valid
        if x_data is None or y_data is None:
            name = dataset_name or file_path.name
            raise ValueError(f"Invalid data in {file_path}: x_data or y_data is None (dataset: {name})")
        
        # Normalize if needed
        if x_data.max() > 1.0:
            x_data = x_data / 255.0
        
        # Reshape if needed
        if len(x_data.shape) == 3:
            x_data = x_data.reshape(-1, input_size, input_size, 1)
        
        return x_data, y_data
    except KeyError as e:
        name = dataset_name or file_path.name
        raise ValueError(f"Missing key '{e}' in {file_path} (dataset: {name})")
    except Exception as e:
        name = dataset_name or file_path.name
        raise ValueError(f"Could not load {name} from {file_path}: {e}")


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
    
    try:
        x_data, y_data = load_dataset_from_npz(npz_path, input_size, 'y', 'font digits')
        print(f"  Loaded font digits ({split}): {len(x_data):,} samples")
        return x_data, y_data
    except FileNotFoundError:
        print(f"Font digits not found at {npz_path}")
        print("Generate with: python DataManagement/FontDigitGenerator.py --api-key YOUR_KEY")
        return None, None


def load_custom_one(split='train', input_size=28):
    """
    Load custom "1" digit variations (serif style with no base).
    
    Args:
        split: 'train' or 'test'
        input_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (x_data, y_data) or (None, None) if not available
    """
    # Try size-specific filename first, then fallback to old format
    npz_path = DATA_DIR / "custom_one" / f"custom_one_{split}_{input_size}x{input_size}_softmax.npz"
    
    try:
        x_data, y_data = load_dataset_from_npz(npz_path, input_size, 'y', 'custom one')
        print(f"  Loaded custom '1' ({split}): {len(x_data):,} samples")
        return x_data, y_data
    except FileNotFoundError:
        print(f"Custom '1' digits not found at {npz_path}")
        print(f"Generate with: python DataManagement/GenerateCustomOne.py --size {input_size}")
        return None, None


def load_and_combine_datasets(input_size=28):
    """
    Load and combine test datasets (all pre-generated at input_size x input_size).
    Training data comes from pre-generated augmented file, so only test data is loaded.
    
    Datasets loaded:
    - EMNIST Digits (pre-generated)
    - ARDIS (pre-generated)
    - USPS (pre-generated)
    - Font Digits (test only)
    - Custom "1" digits (test only)
    - Non-digits (test only)
    
    Args:
        input_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (None, None, x_test, y_test, None)
        - x_train/y_train: Always None (training data from pre-generated augmented file)
        - x_test/y_test: All test data combined
        - is_google_fonts_train: Always None
        Arrays are normalized to [0, 1] and shaped as (samples, input_size, input_size, 1)
    """
    test_datasets = []
    
    print("Loading TEST data only (training data from pre-generated augmented file)...")
    
    # =========================================================================
    # EMNIST Digits
    # =========================================================================
    emnist_test_path = DATA_DIR / "EMNIST" / f"emnist_digits_test_{input_size}x{input_size}.npz"
    
    try:
        x_test, y_test = load_dataset_from_npz(emnist_test_path, input_size, 'y_softmax', 'EMNIST test')
        test_datasets.append((x_test, y_test))
        print(f"  EMNIST: {len(x_test):,} test")
    except FileNotFoundError:
        print(f"  Warning: EMNIST {input_size}x{input_size} not found. Run PregenData{input_size}.py first.")
    
    # =========================================================================
    # ARDIS
    # =========================================================================
    ardis_test_path = DATA_DIR / "ardis" / f"ardis_test_{input_size}x{input_size}.npz"
    
    try:
        x_test, y_test = load_dataset_from_npz(ardis_test_path, input_size, 'y_softmax', 'ARDIS test')
        test_datasets.append((x_test, y_test))
        print(f"  ARDIS: {len(x_test):,} test")
    except FileNotFoundError:
        print(f"  Warning: ARDIS {input_size}x{input_size} not found. Run PregenData{input_size}.py first.")
    
    # =========================================================================
    # USPS
    # =========================================================================
    usps_test_path = DATA_DIR / "usps" / f"usps_test_{input_size}x{input_size}.npz"
    
    try:
        x_test, y_test = load_dataset_from_npz(usps_test_path, input_size, 'y_softmax', 'USPS test')
        test_datasets.append((x_test, y_test))
        print(f"  USPS: {len(x_test):,} test")
    except FileNotFoundError:
        print(f"  Warning: USPS {input_size}x{input_size} not found. Run PregenData{input_size}.py first.")
    
    # =========================================================================
    # Font Digits (test only)
    # =========================================================================
    x_fonts_test, y_fonts_test = load_font_digits(split='test', input_size=input_size)
    if x_fonts_test is not None:
        test_datasets.append((x_fonts_test, y_fonts_test))
        print(f"  Font digits: {len(x_fonts_test):,} test")
    
    # =========================================================================
    # Custom "1" digits (test only)
    # =========================================================================
    x_custom_test, y_custom_test = load_custom_one(split='test', input_size=input_size)
    if x_custom_test is not None:
        test_datasets.append((x_custom_test, y_custom_test))
        print(f"  Custom '1': {len(x_custom_test):,} test")
    
    # =========================================================================
    # Non-digits (test only)
    # =========================================================================
    _, _, x_negative_test, y_negative_test = load_non_digits(image_size=input_size)
    if x_negative_test is not None:
        test_datasets.append((x_negative_test, y_negative_test))
        print(f"  Non-digits: {len(x_negative_test):,} test")
    
    # =========================================================================
    # COMBINE DATASETS
    # =========================================================================
    
    if len(test_datasets) == 0:
        raise ValueError(f"No test datasets found! Run PregenData{input_size}.py to generate {input_size}x{input_size} datasets.")
    
    # Combine test data
    x_test = np.concatenate([ds[0] for ds in test_datasets], axis=0)
    y_test = np.concatenate([ds[1] for ds in test_datasets], axis=0)
    if len(x_test.shape) == 3:
        x_test = x_test.reshape(-1, input_size, input_size, 1)
    
    print(f"\n  Total test: {len(x_test):,} samples")
    return None, None, x_test, y_test, None


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
        # Load PRE-GENERATED augmented data (REQUIRED)
        # =====================================================================
        x_train_aug, y_train_aug = load_augmented_data(image_size=input_size)
        
        if x_train_aug is None:
            raise ValueError(f"Failed to load augmented training data. Run: python3 DataManagement/PregenAugmentedData.py --size {input_size}")
        
        print("\n✓ Using PRE-GENERATED augmented data (fast training mode)")
        x_train_all = x_train_aug
        y_train_all = y_train_aug
        
        # Only load test data (training already in pre-gen file)
        _, _, x_test, y_test, _ = load_and_combine_datasets(input_size=input_size)
        
        # Shuffle test data
        indices = np.random.permutation(len(x_test))
        x_test = x_test[indices]
        y_test = y_test[indices]
        
        print(f"Training samples: {len(x_train_all):,} (pre-augmented)")
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
        - predicted_digit: int (0-9 for digits, 10 for "not a digit")
        - confidence: float (0.0-1.0) - probability of the predicted class
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
    
    # Return the predicted class (0-9 for digits, 10 for non-digit)
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
