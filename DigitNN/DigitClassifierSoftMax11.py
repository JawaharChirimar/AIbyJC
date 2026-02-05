#!/usr/bin/env python3
"""
DigitClassifierSoftMax11.py

Provides functions for creating, training, and using a CNN-based digit classifier.
Uses softmax with 11 classes: 10 digits (0-9) + 1 "not a digit" class (10).
"""

import os
import argparse
import cv2
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from OODDetection import EnergyScorer

_energy_scorer = None
_calibration_data_loaded = False


# Import DATA_DIR from NonDigitGenerator (used for data paths)
from DataManagement.NonDigitGenerator import DATA_DIR
# Import pre-generated augmented data loader
from DataManagement.PregenAugmentedData import load_augmented_data
# Import stratified batch generator
from StratifiedBatchGenerator import create_stratified_batch_generator

# =============================================================================
# CONFIGURABLE CONSTANTS
# =============================================================================
DROPOUT_RATE = 0.5          # Dropout rate in model (prevents overfitting)
BATCH_SIZE = 128             # Batch size for training

class BalancedLoss(keras.losses.Loss):
    """
    Custom loss function that minimizes mean class loss + variance penalty.
    
    This encourages balanced performance across all classes by:
    1. Computing per-class mean losses
    2. Penalizing variance in class losses (higher variance = more imbalance)
    
    Formula: Loss = Mean(Class_Losses) + λ * Variance(Class_Losses)
    
    Where:
    - Mean(Class_Losses) = average cross-entropy loss across all classes
    - Variance(Class_Losses) = mean squared difference from the mean
    - λ = var_penalty parameter (controls strength of variance penalty)
    """
    def __init__(self, num_classes=11, var_penalty=0.5, 
    digit_only_variance=True,
    name='balanced_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
        self.num_classes = num_classes
        self.var_penalty = var_penalty
        self.digit_only_variance = digit_only_variance
        self.ce = keras.losses.SparseCategoricalCrossentropy(reduction='none')
    
    def call(self, y_true, y_pred):
        # Per-sample cross-entropy losses
        sample_losses = self.ce(y_true, y_pred)
        
        # Per-class mean loss
        class_losses = []
        class_counts = []
        for i in range(self.num_classes):
            mask = tf.cast(y_true == i, tf.float32)
            count = tf.reduce_sum(mask)
            class_counts.append(count)
            # Compute loss: sum(sample_losses * mask) / count
            # If count == 0, this will be 0/0 = NaN, which we'll filter out
            class_loss = tf.reduce_sum(sample_losses * mask) / (count + 1e-10)  # Small epsilon to avoid NaN
            class_losses.append(class_loss)
        
        class_losses = tf.stack(class_losses)
        class_counts = tf.stack(class_counts)
        
        # Filter to only include classes present in batch (count > 0)
        # NOTE: When use_stratified=True, both training and validation batches have all classes.
        # When use_stratified=False, batches can have missing classes (defensive handling).
        class_present = class_counts > 0
        valid_class_losses = tf.boolean_mask(class_losses, class_present)
        
        # If no classes present (shouldn't happen), return zero loss
        mean_loss = tf.cond(
            tf.size(valid_class_losses) > 0,
            lambda: tf.reduce_mean(valid_class_losses),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        
        if self.digit_only_variance and self.num_classes == 11:
            digit_losses = class_losses[:10]
            digit_counts = class_counts[:10]
            digit_present = digit_counts > 0
            valid_digit_losses = tf.boolean_mask(digit_losses, digit_present)
            variance = tf.cond(
                tf.size(valid_digit_losses) > 0,
                lambda: tf.reduce_mean(tf.square(valid_digit_losses - tf.reduce_mean(valid_digit_losses))),
                lambda: tf.constant(0.0, dtype=tf.float32))
        else:
            variance = tf.cond(
                tf.size(valid_class_losses) > 0,
                lambda: tf.reduce_mean(tf.square(valid_class_losses - mean_loss)),
                lambda: tf.constant(0.0, dtype=tf.float32))
        
        # Formula: Loss = Mean(Class_Losses) + λ * Variance(Class_Losses)
        penalty = self.var_penalty * variance
        
        return mean_loss + penalty
    
    def get_config(self):
        """Required for saving/loading models with custom loss."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'var_penalty': self.var_penalty,
            'digit_only_variance': self.digit_only_variance,
            'reduction': self.reduction
        })
        return config

def create_digit_classifier_model(input_size=28, use_balanced_loss=False,
lambda_weight=0.5, learning_rate=0.001, neurons_in_dense_layer=64):
    """
    Create a CNN model for digit classification with 11 classes 
    (0-9 digits + 10 "not a digit").
    
    Uses deep model architecture with 6 conv layers, optimized for 
    input_size x input_size images.
    Always uses softmax activation. 
    Loss function can be either BalancedLoss or sparse_categorical_crossentropy.
    
    Args:
        input_size: Image size (28 or 64, default: 28)
        use_balanced_loss: If True, use BalancedLoss for balanced class 
            performance (default: False)
        lambda_weight: Weight for variance penalty in BalancedLoss (default: 0.5)
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        neurons_in_dense_layer: Number of neurons in final dense layer before 
            output (default: 64)
    
    Returns:
        Compiled Keras model
    """
    # Model capacity for large dataset (240k+ samples)
    number_convolution_channels = 32
    number_convolution_channelsF = 64
    
    # Always use softmax with 11 classes (0-9 digits + 10 "not a digit")
    output_activation = 'softmax'
    # Choose loss function
    if use_balanced_loss:
        loss_function = BalancedLoss(num_classes=11, var_penalty=lambda_weight)
    else:
        loss_function = 'sparse_categorical_crossentropy'

    accuracy_metric = 'accuracy'
    
    # Deep model architecture (6 conv layers) for input_size x input_size input
    #conv(number_convolution_channels) → BN → 
    #conv(number_convolution_channels) → BN → pool(2,2) → dropout(0.25) → 
    #conv(number_convolution_channelsF) → BN → 
    #conv(number_convolution_channelsF) → BN → pool(2,2) → dropout(0.25) → 
    #conv(number_convolution_channelsF * 2) → BN →
    #conv(number_convolution_channelsF * 2) → BN → pool(2,2) → dropout(0.25) → 
    #flatten → dense(neurons_in_dense_layer) → BN → 
    #dropout(DROPOUT_RATE) → dense(11) 
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
        layers.Conv2D(number_convolution_channelsF * 2, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(neurons_in_dense_layer, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(11, activation=output_activation)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=[accuracy_metric]
    )
    
    return model


class Softmax11DiagnosticsCallback(keras.callbacks.Callback):
    """
    Callback to print per-epoch diagnostics for softmax 11-class mode.
    Shows per-digit accuracy (0-9) and negative rejection rate (class 10).
    """
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        # Precompute per-class masks for digits 0-9
        self.class_masks = {}
        self.class_labels = {}
        for digit in range(10):
            mask = y_val == digit
            self.class_masks[digit] = mask
            self.class_labels[digit] = y_val[mask] if np.sum(mask) > 0 else None
        # Precompute mask for non-digits (class 10)
        self.negative_mask = y_val == 10
        self.n_negatives = np.sum(self.negative_mask)
    
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        y_pred = self.model.predict(self.x_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        results = []
        per_digit_results = []
        
        # Per-digit accuracy (classes 0-9)
        for digit in range(10):
            mask = self.class_masks[digit]
            if np.sum(mask) > 0:
                digit_acc = np.mean(y_pred_classes[mask] == digit) * 100
                digit_count = np.sum(mask)
                per_digit_results.append(f"{digit}: {digit_acc:.1f}% ({digit_count})")
        
        # Overall digit accuracy (classes 0-9 combined)
        digit_mask = self.y_val < 10
        if np.sum(digit_mask) > 0:
            overall_digit_acc = np.mean(y_pred_classes[digit_mask] == self.y_val[digit_mask]) * 100
            results.append(f"Overall digits: {overall_digit_acc:.1f}%")
        
        # Negative rejection rate (class 10 should be predicted for negatives)
        if self.n_negatives > 0:
            neg_preds = np.argmax(y_pred[self.negative_mask], axis=1)
            neg_rejected = np.sum(neg_preds == 10)
            neg_acc = neg_rejected / self.n_negatives * 100
            results.append(f"Neg rejected: {neg_acc:.1f}% ({neg_rejected}/{self.n_negatives})")
        
        # Print per-digit accuracy
        print(f"  [Softmax11] {' | '.join(results)}")
        print(f"  [Per-digit] {' | '.join(per_digit_results)}")

def calibrate_energy_scorer(classifier_model_path, input_size=28):
    """
    Load a pre-trained digit classifier and calibrate the energy scorer.
    """
    model = load_digit_classifier(classifier_model_path)
    if model is None:
        raise ValueError(f"calibrate_energy_scorer: Could not load model from {classifier_model_path}")
    
    print("Loading test data for energy scorer calibration...")
    _, _, x_test, y_test = load_augmented_data(image_size=input_size)
        
    if x_test is None:
        raise ValueError(f"calibrate_energy_scorer: Failed to load test data")
    else:
        # Filter to only digits (0-9), exclude non-digits (class 10)
        digit_mask = y_test < 10
        x_test_digits = x_test[digit_mask]
            
    _energy_scorer = EnergyScorer(model)
    thresholds = _energy_scorer.calibrate(
        x_digits=x_test_digits, 
        percentile=[99.9, 99.5,99, 95, 90])
    
    # Create dictionary mapping percentiles to thresholds
    percentiles = [99.9, 99.5, 99, 95, 90]
    thresholds_dict = {str(p): float(t) for p, t in zip(percentiles, thresholds)}
    
    # Create calibration data with thresholds and timestamp
    calibration_data = {
        'thresholds': thresholds_dict,
        'time': datetime.now().isoformat()
    }
    
    # Derive energy calibration file path from model path
    # e.g., /path/to/xy.keras -> /path/to/energy_xy_calibrate.json
    energy_file_path = EnergyScorer.calibration_file_name_from_model_path(classifier_model_path)
    
    # Write to JSON file (overwrite if exists)
    try:
        with open(energy_file_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Energy calibration saved to {energy_file_path}")
        return True
    except Exception as e:
        raise ValueError(f"calibrate_energy_scorer: Failed to save calibration to {energy_file_path}: {e}")

def load_digit_classifier(classifier_model_path):
    """
    Load a pre-trained digit classifier.
    """
    if classifier_model_path is None or classifier_model_path == '' or not classifier_model_path:
        raise ValueError("load_digit_classifier:classifier_model_path must be provided when train_model=False")
    
    classifier_model_path = str(classifier_model_path)  # Convert Path to string if needed
    if not os.path.exists(classifier_model_path):
        raise ValueError(f"load_digit_classifier:Model file not found: {classifier_model_path}.")
    
    try:
        print(f"Loading digit classifier from: {classifier_model_path}")
        # Provide custom objects in case model was saved with BalancedLoss
        model = keras.models.load_model(
            classifier_model_path,
            custom_objects={'BalancedLoss': BalancedLoss}
        )
        print("Digit classifier loaded successfully")

        # Try to load calibration if it exists (optional)
        global _energy_scorer, _calibration_data_loaded
        if _energy_scorer is None:
            _energy_scorer = EnergyScorer(model)
            calibration_file = _energy_scorer.calibration_file_name_from_model_path(classifier_model_path)
            if os.path.exists(calibration_file):
                print("Loading energy scorer calibration for model...")
                try:
                    _energy_scorer.load_calibration(classifier_model_path, percentile=99.9)
                    _calibration_data_loaded = True
                except Exception as e:
                    print(f"Warning: Could not load calibration: {e}")
            else:
                print("No calibration file found - calibration will need to be done separately")

        return model  
    except Exception as e:
        raise ValueError(f"load_digit_classifier:Cannot load model from {classifier_model_path}: {e}. Set train_model=True to create a new model.")

    return None


def train_digit_classifier(
    num_epochs=20, input_size=28, initial_model_path=None,
    use_balanced_loss=False, lambda_weight=0.5, learning_rate=0.001, 
    neurons_in_dense_layer=64):
    """
    Load a pre-trained digit classifier or create/train a new one.
    
    Always uses pre-generated augmented data and 6 conv layer CNN model.
    Always uses stratified batch sampling for balanced training.

    Args:
        num_epochs: Number of training epochs (default: 20)
        input_size: Image size (28 or 64, default: 28)
        initial_model_path: Path to pre-trained model to use as starting point (optional)
        use_balanced_loss: If True, use BalancedLoss for balanced class performance (default: False)
        lambda_weight: Weight for variance penalty in BalancedLoss (default: 0.5)
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        neurons_in_dense_layer: Number of neurons in final dense layer before output (default: 64)
        Always uses softmax with 11 classes (0-9 digits + 10 "not a digit")
    
    Returns:
        Trained Keras model for digit classification
    """

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
            # Provide custom objects in case model was saved with BalancedLoss
            model = keras.models.load_model(
                initial_model_path,
                custom_objects={'BalancedLoss': BalancedLoss}
            )
            print(f"  Initial model loaded successfully")
            # Recompile with new learning rate and loss function settings
            if use_balanced_loss:
                loss_function = BalancedLoss(num_classes=11, 
                var_penalty=lambda_weight,
                digit_only_variance=True)
            else:
                loss_function = 'sparse_categorical_crossentropy'
                
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss_function,
                metrics=['accuracy']
            )

            print(f"  Model recompiled with learning rate: {learning_rate}")
            if use_balanced_loss:
                print(f"  Model using balanced loss with lambda weight: {lambda_weight}")
                print(f"  Model using balanced loss digit only variance")

        except Exception as e:
            print(f"  Warning: Could not load initial model: {e}")
            print(f"  Creating new model instead...")
            model = create_digit_classifier_model(
                input_size=input_size,
                use_balanced_loss=use_balanced_loss,
                lambda_weight=lambda_weight,
                learning_rate=learning_rate,
                neurons_in_dense_layer=neurons_in_dense_layer
            )
    else:
        # Create new model (always uses softmax with 11 classes)
        if initial_model_path is not None:
            print(f"Warning: Initial model path provided but file not found: {initial_model_path}")
            print(f"  Creating new model instead...")
        else:
            print(f"Creating new digit classifier model ({input_size}x{input_size} input, 11 classes)...")
        model = create_digit_classifier_model(
            input_size=input_size,
            use_balanced_loss=use_balanced_loss,
            lambda_weight=lambda_weight,
            learning_rate=learning_rate,
            neurons_in_dense_layer=neurons_in_dense_layer
        )
    
    # Try to train on all digit datasets
    try:
        # =====================================================================
        # Load PRE-GENERATED augmented data (REQUIRED)
        # =====================================================================
        x_train_all, y_train_all, x_test, y_test = load_augmented_data(image_size=input_size)
        
        if x_train_all is None:
            raise ValueError(f"Failed to load augmented data. Run augmentation scripts for size {input_size}")
        
        print("\n✓ Using PRE-GENERATED augmented data (fast training mode)")
        
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
        
        print("\nUsing STRATIFIED batch sampling (balanced classes per batch)")
        # Create the generator function (returns a callable that returns a generator)
        train_generator = create_stratified_batch_generator(
            x_train_all, y_train_all, batch_size=batch_size, num_classes=11, non_digit_class=10
        )
        
        # Create tf.data.Dataset from generator (NO memory copy!)
        train_dataset = tf.data.Dataset.from_generator(
            train_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, input_size, input_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset - use from_tensor_slices for validation 
        #                      (references data, no copy)
        # Validation is small enough (~90K samples = ~1.4GB) that keeping in 
        # memory is acceptable
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
                patience=100, #patience,
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
            Softmax11DiagnosticsCallback(x_test, y_test),
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
        
        # Save the final model 
        # (also saved by checkpoint, but this ensures final state is saved)
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
        digit_image: Greyscale image (numpy array), will be resized to 
            input_size x input_size if needed
        input_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (predicted_digit, confidence)
        - predicted_digit: int (0-9 for digits, 10 for "not a digit")
        - confidence: float (0.0-1.0) - probability of the predicted class
    """
    # Ensure image is the right shape and type
    if digit_image.shape != (input_size, input_size):
        # Resize using LANCZOS for quality
        digit_image = cv2.resize(
            digit_image, 
            (input_size, input_size), 
            interpolation=cv2.INTER_LANCZOS4
        )
    
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
    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

    global _energy_scorer, _calibration_data_loaded
    
    # Calibrate is loaded when classifer model is loaded
    if _calibration_data_loaded:
        score = _energy_scorer.score(digit_input)
        print(f"Energy score: {score}")
        # Use energy score to detect OOD: if energy > threshold, reject as non-digit
        is_ood = _energy_scorer.is_ood(digit_input)
        if is_ood and predicted_class != 10:
            # Energy indicates OOD - override prediction to non-digit (class 10)
            # even if classifier predicted a digit
            predicted_class = 11
            confidence = float(predictions[0][10])  # Use classifier's confidence for class 10
            print(f"  Energy-based OOD detection: Overriding to class 11 (non-digit)")
        else:
            print(f"  Energy-based OOD detection: Predicted class {predicted_class} is not OOD")
    else:
        score = None
        print(f"  Energy scorer not calibrated, using default threshold")
    
    # Return the predicted class (0-9 for digits, 10 for non-digit)
    return predicted_class, confidence, score


def main():
    """
    Standalone training function for the digit classifier.
    """
    
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
        "--energy-calibrate",
        action="store_true",
        help="True means calibrate energy model using the specified model"
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
    parser.add_argument(
        "--balanced-loss",
        action="store_true",
        help="Use BalancedLoss in training (optimizes for balanced performance during backpropagation)"
    )
    parser.add_argument(
        "--lambda-weight",
        type=float,
        default=0.5,
        help="Weight for variance penalty in BalancedLoss (default: 0.5)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001)"
    )
    parser.add_argument(
        "--dense-layer",
        type=int,
        default=32,
        help="Number of neurons in final dense layer (default: 32)"
    )
    
    args = parser.parse_args()
    
    # Determine input size
    input_size = args.size
        
    if args.train_model and args.energy_calibrate:
        print(f"Please specify only option. Either --train-model or --energy-calibrate, not both.")
        return 0

    if args.energy_calibrate:
        print(f"Starting energy model calibration...")
        print(f"Trained classifier model path: {args.model_path}")
        print(f"Input image size: {input_size}x{input_size}")
        calibrate_energy_scorer(args.model_path, input_size=input_size)
        return 1

    if args.train_model:
        print(f"Starting digit classifier training with 11 classes (0-9 digits + 10 'not a digit')...")
        print(f"Input image size: {input_size}x{input_size}")
        print(f"Epochs: {args.epoch_count}")
        print(f"Initial model: {args.initial_model}")
        print(f"Balanced loss (training): {'ENABLED' if args.balanced_loss else 'DISABLED'}")
        if args.balanced_loss:
            print(f"  Lambda weight: {args.lambda_weight}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Dense layer neurons: {args.dense_layer}")
        # Train the model (always uses softmax with 11 classes)
        model = train_digit_classifier(
            num_epochs=args.epoch_count,
            input_size=input_size,
            initial_model_path=args.initial_model,
            use_balanced_loss=args.balanced_loss,
            lambda_weight=args.lambda_weight,
            learning_rate=args.learning_rate,
            neurons_in_dense_layer=args.dense_layer
        )
        print("\nTraining complete!")
        return 2

    # Default case: no options specified
    print(f"Please specify one of the following options:")
    print(f"  --train-model to train a new model")
    print(f"  --energy-calibrate to load an existing model")
    return 3



if __name__ == "__main__":
    main()
