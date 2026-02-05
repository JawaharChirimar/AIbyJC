#!/usr/bin/env python3
"""
DigitClassifierSoftMax10.py

Provides functions for creating, training, and using a CNN-based digit classifier.
Uses softmax with 10 classes: digits 0-9 only (no non-digit class).

Supports on-the-fly data loading and augmentation with optional datasets:
- MNIST (base, always required)
- USPS (optional, --get_USPS flag)
- ARDIS (optional, --get_ARDIS flag)
- Custom One (optional, --get_CustomONE flag)
- Data augmentation (optional, --augment flag)
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

# Data directory
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"

# =============================================================================
# CONFIGURABLE CONSTANTS
# =============================================================================
DROPOUT_RATE = 0.25          # Dropout rate in model (prevents overfitting)
BATCH_SIZE = 64             # Batch size for training


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
    def __init__(self, num_classes=10, var_penalty=0.5, name='balanced_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
        self.num_classes = num_classes
        self.var_penalty = var_penalty
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
        
        variance = tf.cond(
            tf.size(valid_class_losses) > 0,
            lambda: tf.reduce_mean(tf.square(valid_class_losses - mean_loss)),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        
        # Formula: Loss = Mean(Class_Losses) + λ * Variance(Class_Losses)
        penalty = self.var_penalty * variance
        
        return mean_loss + penalty
    
    def get_config(self):
        """Required for saving/loading models with custom loss."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'var_penalty': self.var_penalty,
            'reduction': self.reduction
        })
        return config


def create_digit_classifier_model(input_size=28, use_balanced_loss=False, lambda_weight=0.5, learning_rate=0.001, neurons_in_dense_layer=32):
    """
    Create a CNN model for digit classification with 10 classes (digits 0-9 only).
    
    Uses deep model architecture with 4 conv layers, optimized for input_size x input_size images.
    Always uses softmax activation.
    
    Args:
        input_size: Image size (28 or 64, default: 28)
        use_balanced_loss: Use BalancedLoss instead of cross-entropy (default: False)
        lambda_weight: Variance penalty weight for BalancedLoss (default: 0.5)
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        neurons_in_dense_layer: Number of neurons in final dense layer (default: 32)
    
    Returns:
        Compiled Keras model
    """
    # Model capacity for large dataset (90K to 120K samples)
    number_convolution_channels = 32
    number_convolution_channelsF = 64 
    
    # Always use softmax with 10 classes (0-9 digits only)
    output_activation = 'softmax'
    
    # Choose loss function
    if use_balanced_loss:
        loss_function = BalancedLoss(num_classes=10, var_penalty=lambda_weight)
    else:
        loss_function = 'sparse_categorical_crossentropy'
    
    accuracy_metric = 'accuracy'
    
    # Deep model architecture (3 conv layers) for input_size x input_size input
    #conv(number_convolution_channels) → BN → pool(2,2) → 
    #conv(number_convolution_channels) → BN → pool(2,2) → dropout(0.25) → 
    #conv(number_convolution_channelsF) → BN → pool(2,2) → dropout(0.25) → 
    #flatten → dense(neurons_in_dense_layer) → BN → dropout(0.5) → dense(10) 
    model = keras.Sequential([
        layers.Input(shape=(input_size, input_size, 1)),
        layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(number_convolution_channelsF, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(neurons_in_dense_layer, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(10, activation=output_activation)  # 10 classes: digits 0-9 only
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
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


class Softmax10DiagnosticsCallback(keras.callbacks.Callback):
    """
    Callback to print per-epoch diagnostics for softmax 10-class mode.
    Shows per-digit accuracy (0-9) and computes balanced score metric.
    """
    def __init__(self, x_val, y_val, lambda_weight=0.5):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.lambda_weight = lambda_weight  # Weight for variance penalty
        # Precompute per-class masks
        self.class_masks = {}
        self.class_labels = {}
        for digit in range(10):
            mask = y_val == digit
            self.class_masks[digit] = mask
            self.class_labels[digit] = y_val[mask] if np.sum(mask) > 0 else None
    
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        y_pred = self.model.predict(self.x_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        results = []
        per_class_accuracies = []
        
        # Per-digit accuracy (classes 0-9)
        for digit in range(10):
            mask = self.class_masks[digit]
            if np.sum(mask) > 0:
                digit_acc = np.mean(y_pred_classes[mask] == digit) * 100
                digit_count = np.sum(mask)
                results.append(f"{digit}: {digit_acc:.1f}% ({digit_count})")
                per_class_accuracies.append(digit_acc / 100.0)  # Convert to [0, 1] range
            else:
                # If class has no samples, skip it (shouldn't happen, but handle gracefully)
                per_class_accuracies.append(0.0)
        
        # Overall accuracy
        overall_acc = np.mean(y_pred_classes == self.y_val) * 100
        
        # Compute composite metric: Mean(VA) - λ * Variance(VA)
        if len(per_class_accuracies) == 10:
            mean_accuracy = np.mean(per_class_accuracies)
            variance = np.var(per_class_accuracies)  # This is (Σ(VA(i) - Mean(VA))²) / 10
            balanced_score = mean_accuracy - self.lambda_weight * variance
            
            # Add to logs so it can be monitored by EarlyStopping/ModelCheckpoint
            if logs is not None:
                logs['val_balanced_score'] = balanced_score
                logs['val_mean_class_accuracy'] = mean_accuracy
                logs['val_class_variance'] = variance
            
            print(f"  [Softmax10] Overall: {overall_acc:.1f}% | Per-digit: {' | '.join(results)}")
            print(f"  [Balanced] Mean: {mean_accuracy*100:.2f}% | Variance: {variance*10000:.2f} (×10⁻⁴) | Score: {balanced_score*100:.2f}%")
        else:
            print(f"  [Softmax10] Overall: {overall_acc:.1f}% | Per-digit: {' | '.join(results)}")


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


def load_custom_one(split='train', input_size=28):
    """
    Load custom "1" digit variations (serif style with no base).
    
    Args:
        split: 'train' or 'test'
        input_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (x_data, y_data) or (None, None) if not available
    """
    npz_path = DATA_DIR / "custom_one" / f"custom_one_{split}_{input_size}x{input_size}_softmax.npz"
    
    try:
        x_data, y_data = load_dataset_from_npz(npz_path, input_size, 'y', 'custom one')
        print(f"  Loaded custom '1' ({split}): {len(x_data):,} samples")
        return x_data, y_data
    except FileNotFoundError:
        print(f"Custom '1' digits not found at {npz_path}")
        print(f"Generate with: python DataManagement/GenerateCustomOne.py --size {input_size}")
        return None, None


def load_and_combine_datasets(input_size=28, use_mnist=True, use_usps=False, use_ardis=False, use_custom_one=False):
    """
    Load and combine digit datasets (all pre-generated at input_size x input_size):
    - MNIST (base, always required if use_mnist=True)
    - USPS (optional, if use_usps=True)
    - ARDIS (optional, if use_ardis=True)
    - Custom "1" digits (optional, if use_custom_one=True)
    
    Args:
        input_size: Image size (28 or 64, default: 28)
        use_mnist: Load MNIST dataset (default: True, always required)
        use_usps: Load USPS dataset (default: False)
        use_ardis: Load ARDIS dataset (default: False)
        use_custom_one: Load Custom One dataset (default: False)
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test, is_google_fonts_train)
        - x_train/y_train: All training data combined
        - x_test/y_test: All test data combined
        - is_google_fonts_train: Boolean array (all False, no Google Fonts)
        Arrays are normalized to [0, 1] and shaped as (samples, input_size, input_size, 1)
    """
    train_datasets = []
    test_datasets = []
    train_names = []
    
    print(f"Loading {input_size}x{input_size} datasets...")
    
    # =========================================================================
    # MNIST (base, always required)
    # =========================================================================
    if use_mnist:
        mnist_train_path = DATA_DIR / "MNIST" / f"mnist_train_{input_size}x{input_size}.npz"
        mnist_test_path = DATA_DIR / "MNIST" / f"mnist_test_{input_size}x{input_size}.npz"
        
        try:
            x_train, y_train = load_dataset_from_npz(mnist_train_path, input_size, 'y_softmax', 'MNIST train')
            x_test, y_test = load_dataset_from_npz(mnist_test_path, input_size, 'y_softmax', 'MNIST test')
            train_datasets.append((x_train, y_train))
            test_datasets.append((x_test, y_test))
            train_names.append(f"MNIST ({len(x_train):,})")
            print(f"  MNIST: {len(x_train):,} train, {len(x_test):,} test")
        except FileNotFoundError:
            raise ValueError(f"MNIST {input_size}x{input_size} not found. Run PregenData.py --size {input_size} first.")
        except Exception as e:
            raise ValueError(f"Could not load MNIST {input_size}x{input_size}: {e}")
    
    # =========================================================================
    # USPS (optional)
    # =========================================================================
    if use_usps:
        usps_train_path = DATA_DIR / "usps" / f"usps_train_{input_size}x{input_size}.npz"
        usps_test_path = DATA_DIR / "usps" / f"usps_test_{input_size}x{input_size}.npz"
        
        try:
            x_train, y_train = load_dataset_from_npz(usps_train_path, input_size, 'y_softmax', 'USPS train')
            x_test, y_test = load_dataset_from_npz(usps_test_path, input_size, 'y_softmax', 'USPS test')
            train_datasets.append((x_train, y_train))
            test_datasets.append((x_test, y_test))
            train_names.append(f"USPS ({len(x_train):,})")
            print(f"  USPS: {len(x_train):,} train, {len(x_test):,} test")
        except FileNotFoundError:
            print(f"  Warning: USPS {input_size}x{input_size} not found. Run PregenData.py --size {input_size} first.")
        except Exception as e:
            print(f"  Warning: Could not load USPS {input_size}x{input_size}: {e}")
    
    # =========================================================================
    # ARDIS (optional)
    # =========================================================================
    if use_ardis:
        ardis_train_path = DATA_DIR / "ardis" / f"ardis_train_{input_size}x{input_size}.npz"
        ardis_test_path = DATA_DIR / "ardis" / f"ardis_test_{input_size}x{input_size}.npz"
        
        try:
            x_train, y_train = load_dataset_from_npz(ardis_train_path, input_size, 'y_softmax', 'ARDIS train')
            x_test, y_test = load_dataset_from_npz(ardis_test_path, input_size, 'y_softmax', 'ARDIS test')
            train_datasets.append((x_train, y_train))
            test_datasets.append((x_test, y_test))
            train_names.append(f"ARDIS ({len(x_train):,})")
            print(f"  ARDIS: {len(x_train):,} train, {len(x_test):,} test")
        except FileNotFoundError:
            print(f"  Warning: ARDIS {input_size}x{input_size} not found. Run PregenData.py --size {input_size} first.")
        except Exception as e:
            print(f"  Warning: Could not load ARDIS {input_size}x{input_size}: {e}")
    
    # =========================================================================
    # Custom "1" digits (optional)
    # =========================================================================
    if use_custom_one:
        print("\nLoading custom '1' digits...")
        x_custom_train, y_custom_train = load_custom_one(split='train', input_size=input_size)
        if x_custom_train is not None:
            train_datasets.append((x_custom_train, y_custom_train))
            train_names.append(f"CustomOne ({len(x_custom_train):,})")
        
        x_custom_test, y_custom_test = load_custom_one(split='test', input_size=input_size)
        if x_custom_test is not None:
            test_datasets.append((x_custom_test, y_custom_test))
    
    # =========================================================================
    # COMBINE DATASETS
    # =========================================================================
    
    if len(test_datasets) == 0:
        raise ValueError(f"No test datasets found! Run PregenData.py --size {input_size} to generate {input_size}x{input_size} datasets.")
    
    # Combine test data
    x_test = np.concatenate([ds[0] for ds in test_datasets], axis=0)
    y_test = np.concatenate([ds[1] for ds in test_datasets], axis=0)
    if len(x_test.shape) == 3:
        x_test = x_test.reshape(-1, input_size, input_size, 1)
    
    # Combine training data
    if len(train_datasets) == 0:
        raise ValueError(f"No training datasets found! Run PregenData.py --size {input_size} to generate {input_size}x{input_size} datasets.")
    
    x_train = np.concatenate([ds[0] for ds in train_datasets], axis=0)
    y_train = np.concatenate([ds[1] for ds in train_datasets], axis=0)
    # No Google Fonts in SoftMax10, so is_google_fonts_train is always all False
    is_google_fonts_train = np.zeros(len(x_train), dtype=bool)
    
    if len(x_train.shape) == 3:
        x_train = x_train.reshape(-1, input_size, input_size, 1)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset Summary (all {input_size}x{input_size})")
    print(f"{'='*60}")
    print(f"Training datasets: {' + '.join(train_names)}")
    print(f"  Total training: {len(x_train):,} samples")
    print(f"Test set: {len(x_test):,} samples")
    print(f"{'='*60}\n")
    
    return x_train, y_train, x_test, y_test, is_google_fonts_train


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

        return model  
    except Exception as e:
        raise ValueError(f"load_digit_classifier:Cannot load model from {classifier_model_path}: {e}. Set train_model=True to create a new model.")

    return None

def load_or_create_digit_classifier( 
    num_epochs=20, use_stratified=False, input_size=28, initial_model_path=None,
    use_mnist=True, use_usps=False, use_ardis=False, use_custom_one=False, use_augment=False,
    use_balanced_loss=False, lambda_weight=0.5, learning_rate=0.001, neurons_in_dense_layer=32):
    """
    Load a pre-trained digit classifier or create/train a new one.
    
    Uses MNIST as base dataset with optional additional datasets and augmentation.
    
    Args:
        num_epochs: Number of training epochs (default: 20)
        use_stratified: Use stratified batch sampling (default: False)
        input_size: Image size (28 or 64, default: 28)
        initial_model_path: Path to pre-trained model to use as starting point (optional)
        use_mnist: Load MNIST dataset (default: True, always required)
        use_usps: Load USPS dataset (default: False)
        use_ardis: Load ARDIS dataset (default: False)
        use_custom_one: Load Custom One dataset (default: False)
        use_augment: Enable data augmentation (default: False)
        use_balanced_loss: Use BalancedLoss in training (optimizes for balanced performance during backprop) (default: False)
        lambda_weight: Weight for variance penalty in BalancedLoss (default: 0.5)
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        neurons_in_dense_layer: Number of neurons in final dense layer (default: 32)
        Always uses softmax with 10 classes (digits 0-9 only)
    
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
        print(f"  Learning rate: {learning_rate}")
        print(f"  Dense layer neurons: {neurons_in_dense_layer}")
        try:
            # Include BalancedLoss in custom_objects in case model was saved with it
            custom_objects = {'BalancedLoss': BalancedLoss}
            model = keras.models.load_model(
                initial_model_path, 
                custom_objects=custom_objects)
            print(f"  Initial model loaded successfully")
            # Recompile with new learning rate and loss function settings
            if use_balanced_loss:
                loss_function = BalancedLoss(num_classes=10, var_penalty=lambda_weight)
            else:
                loss_function = 'sparse_categorical_crossentropy'
                
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss_function,
                metrics=['accuracy']
            )
            print(f"  Model recompiled with learning rate: {learning_rate}")
        except Exception as e:
            print(f"  Warning: Could not load initial model: {e}")
            print(f"  Creating new model instead...")
            model = create_digit_classifier_model(input_size=input_size, use_balanced_loss=use_balanced_loss, lambda_weight=lambda_weight, learning_rate=learning_rate, neurons_in_dense_layer=neurons_in_dense_layer)
    else:
        # Create new model (always uses softmax with 10 classes)
        if initial_model_path is not None:
            print(f"Warning: Initial model path provided but file not found: {initial_model_path}")
            print(f"  Creating new model instead...")
        else:
            loss_type = "BalancedLoss" if use_balanced_loss else "cross-entropy"
            print(f"Creating new digit classifier model ({input_size}x{input_size} input, 10 classes, loss: {loss_type})...")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Dense layer neurons: {neurons_in_dense_layer}")
        model = create_digit_classifier_model(input_size=input_size, use_balanced_loss=use_balanced_loss, lambda_weight=lambda_weight, learning_rate=learning_rate, neurons_in_dense_layer=neurons_in_dense_layer)
    
    # Load datasets based on flags
    try:
        # Load and combine datasets
        x_train, y_train, x_test, y_test, is_google_fonts_train = load_and_combine_datasets(
            input_size=input_size,
            use_mnist=use_mnist,
            use_usps=use_usps,
            use_ardis=use_ardis,
            use_custom_one=use_custom_one
        )
        
        # Print per-class distribution for digits (0-9)
        print(f"\n=== Per-Class Distribution (Training) ===")
        unique_digits, digit_counts = np.unique(y_train, return_counts=True)
        total_digits = len(y_train)
        for digit, count in zip(unique_digits, digit_counts):
            if digit < 10:  # Only digits 0-9
                print(f"  Digit {digit}: {count:,} samples ({count/total_digits*100:.1f}% of digits)")
        print(f"==========================================\n")
        
        print(f"Training samples: {len(x_train):,}")
        print(f"Test samples: {len(x_test):,}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Data augmentation: {'ENABLED' if use_augment else 'DISABLED'}")
        
        # =====================================================================
        # TRAINING SETUP
        # =====================================================================
        import time
        
        batch_size = BATCH_SIZE
        print(f"Epoch models will be saved as: {run_dir}/digit_classifier_epoch_XX.keras")
        
        if use_augment:
            # =================================================================
            # AUGMENTATION MODE: Generate augmented data once, keep in memory
            # =================================================================
            print("\n" + "="*60)
            print("AUGMENTATION MODE: Generating augmented data once...")
            print("="*60)
            print("  - 10% of each class selected for augmentation")
            print("  - Each augmented sample → 6 images (original + 5 transforms)")
            print("  - Transforms: rotation, shear+, shear-, aspect wide, aspect narrow")
            print("  - Post-processing: 20% blur, 10% thin, 10% thick, 10% erasure, 10% breaks")
            print("="*60 + "\n")
            
            from collections import Counter
            
            # If stratified, sample balanced subset BEFORE augmentation
            if use_stratified:
                print("\n" + "="*60)
                print("STRATIFIED + AUGMENTATION: Sampling balanced subset first...")
                print("="*60)
                
                # Calculate min_class_count for training
                train_class_counts = Counter(y_train)
                min_class_count_train = min(train_class_counts.values())
                print(f"  Training: min_class_count = {min_class_count_train:,} per class")
                
                # Randomly sample min_class_count from each class for training
                train_balanced_indices = []
                for class_idx in range(10):
                    class_indices = np.where(y_train == class_idx)[0]
                    if len(class_indices) > min_class_count_train:
                        sampled = np.random.choice(class_indices, size=min_class_count_train, replace=False)
                    else:
                        sampled = class_indices
                    train_balanced_indices.extend(sampled)
                
                train_balanced_indices = np.array(train_balanced_indices)
                x_train_balanced = x_train[train_balanced_indices]
                y_train_balanced = y_train[train_balanced_indices]
                is_google_fonts_train_balanced = is_google_fonts_train[train_balanced_indices] if is_google_fonts_train is not None else None
                
                print(f"  Training: {len(x_train_balanced):,} samples ({min_class_count_train:,} per class)")
                
                # Calculate min_class_count for validation
                test_class_counts = Counter(y_test)
                min_class_count_test = min(test_class_counts.values())
                print(f"  Validation: min_class_count = {min_class_count_test:,} per class")
                
                # Randomly sample min_class_count from each class for validation
                test_balanced_indices = []
                for class_idx in range(10):
                    class_indices = np.where(y_test == class_idx)[0]
                    if len(class_indices) > min_class_count_test:
                        sampled = np.random.choice(class_indices, size=min_class_count_test, replace=False)
                    else:
                        sampled = class_indices
                    test_balanced_indices.extend(sampled)
                
                test_balanced_indices = np.array(test_balanced_indices)
                x_test_balanced = x_test[test_balanced_indices]
                y_test_balanced = y_test[test_balanced_indices]
                
                print(f"  Validation: {len(x_test_balanced):,} samples ({min_class_count_test:,} per class)")
                print("="*60 + "\n")
                
                # Use balanced subsets for augmentation
                x_train_for_aug = x_train_balanced
                y_train_for_aug = y_train_balanced
                is_google_fonts_train_for_aug = is_google_fonts_train_balanced
                x_test_for_aug = x_test_balanced
                y_test_for_aug = y_test_balanced
            else:
                # Non-stratified: use all samples for augmentation
                x_train_for_aug = x_train
                y_train_for_aug = y_train
                is_google_fonts_train_for_aug = is_google_fonts_train
                x_test_for_aug = x_test
                y_test_for_aug = y_test
            
            # Create data augmentation generator (from DataAugmentation.py)
            print("Generating augmented data (one pass)...")
            train_datagen = ImageDataGeneratorWithAugmentation(
                x_train_for_aug, 
                y_train_for_aug,
                is_google_fonts=is_google_fonts_train_for_aug,
                augment_ratio=0.10,  # 10% selected for augmentation (matching PregenAugmentedData.py)
                num_classes=10,  # 10 classes only (0-9)
                non_digit_class=None  # No non-digit class
            )
            
            # Generate all augmented data once (process all samples in one batch)
            start_time = time.time()
            augmented_x = []
            augmented_y = []
            
            # Process exactly one epoch (one batch with all samples)
            gen_batch_size = len(x_train_for_aug)  # Process all at once
            for batch_x, batch_y in train_datagen.flow(batch_size=gen_batch_size, shuffle=True):
                augmented_x.append(batch_x)
                augmented_y.append(batch_y)
                break  # Only one batch needed since batch_size = len(x_train_for_aug)
            
            # Concatenate all batches
            print("Concatenating augmented data...")
            x_train_aug = np.concatenate(augmented_x, axis=0)
            y_train_aug = np.concatenate(augmented_y, axis=0)
            
            # Shuffle
            print("Shuffling augmented data...")
            perm = np.random.permutation(len(x_train_aug))
            x_train_aug = x_train_aug[perm]
            y_train_aug = y_train_aug[perm]
            
            gen_time = time.time() - start_time
            print(f"\n✓ Augmented data generated in {gen_time:.1f}s")
            print(f"  Original samples: {len(x_train_for_aug):,}")
            print(f"  Augmented samples: {len(x_train_aug):,}")
            print(f"  Expansion factor: {len(x_train_aug)/len(x_train_for_aug):.2f}x")
            
            # Print augmentation statistics
            train_datagen.print_epoch_stats(epoch_num=1)
            
            # Use augmented data for training (same as no-augmentation mode)
            x_train = x_train_aug
            y_train = y_train_aug
            print(f"\nUsing augmented data for training (same across all epochs)")
            
            # =================================================================
            # AUGMENT VALIDATION DATA (same approach as training)
            # =================================================================
            print("\n" + "="*60)
            print("AUGMENTING VALIDATION DATA...")
            print("="*60)
            
            # Create data augmentation generator for validation data
            # No Google Fonts in SoftMax10, so is_google_fonts_test is all False
            is_google_fonts_test_for_aug = np.zeros(len(x_test_for_aug), dtype=bool)
            
            val_datagen = ImageDataGeneratorWithAugmentation(
                x_test_for_aug, 
                y_test_for_aug,
                is_google_fonts=is_google_fonts_test_for_aug,
                augment_ratio=0.10,  # 10% selected for augmentation (matching training)
                num_classes=10,  # 10 classes only (0-9)
                non_digit_class=None  # No non-digit class
            )
            
            # Generate all augmented validation data once (process all samples in one batch)
            start_time = time.time()
            augmented_val_x = []
            augmented_val_y = []
            
            # Process exactly one epoch (one batch with all samples)
            gen_batch_size = len(x_test_for_aug)  # Process all at once
            for batch_x, batch_y in val_datagen.flow(batch_size=gen_batch_size, shuffle=True):
                augmented_val_x.append(batch_x)
                augmented_val_y.append(batch_y)
                break  # Only one batch needed since batch_size = len(x_test_for_aug)
            
            # Concatenate all batches
            print("Concatenating augmented validation data...")
            x_test_aug = np.concatenate(augmented_val_x, axis=0)
            y_test_aug = np.concatenate(augmented_val_y, axis=0)
            
            # Shuffle
            print("Shuffling augmented validation data...")
            perm = np.random.permutation(len(x_test_aug))
            x_test_aug = x_test_aug[perm]
            y_test_aug = y_test_aug[perm]
            
            gen_time = time.time() - start_time
            print(f"\n✓ Augmented validation data generated in {gen_time:.1f}s")
            print(f"  Original validation samples: {len(x_test_for_aug):,}")
            print(f"  Augmented validation samples: {len(x_test_aug):,}")
            print(f"  Expansion factor: {len(x_test_aug)/len(x_test_for_aug):.2f}x")
            
            # Print augmentation statistics
            val_datagen.print_epoch_stats(epoch_num=1)
            
            # Use augmented validation data
            x_test = x_test_aug
            y_test = y_test_aug
            print(f"\nUsing augmented validation data (same across all epochs)")
            print("="*60 + "\n")
        
        # =================================================================
        # TRAINING: Standard training with model.fit() (used by both modes)
        # =================================================================
        if not use_augment:
            print("\n" + "="*60)
            print("STANDARD TRAINING MODE (no augmentation)")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("TRAINING WITH PRE-GENERATED AUGMENTED DATA")
            print("="*60)
        
        # Choose generator based on stratified flag
        balanced_samples_per_epoch = None  # Initialize for later use
        if use_stratified:
            print("\nUsing STRATIFIED batch sampling (balanced classes per batch)")
            from StratifiedBatchGenerator import create_stratified_batch_generator
            from collections import Counter
            # Create the generator function (returns a callable that returns a generator)
            train_generator = create_stratified_batch_generator(
                x_train, y_train, batch_size=batch_size, num_classes=10, non_digit_class=None
            )
            # Calculate steps_per_epoch for balanced sampling
            class_counts = Counter(y_train)
            min_class_count = min(class_counts.values())
            balanced_samples_per_epoch = min_class_count * 10  # 10 classes
            steps_per_epoch = balanced_samples_per_epoch // batch_size
            print(f"  Balanced training: {min_class_count:,} samples per class = {balanced_samples_per_epoch:,} total")
            print(f"  Steps per epoch: {steps_per_epoch}")
        else:
            steps_per_epoch = len(x_train) // batch_size
            # Standard random shuffle generator
            def train_generator():
                n_samples = len(x_train)
                indices = np.arange(n_samples)
                while True:
                    np.random.shuffle(indices)
                    for start in range(0, n_samples, batch_size):
                        end = min(start + batch_size, n_samples)
                        batch_indices = indices[start:end]
                        yield x_train[batch_indices], y_train[batch_indices]
        
        val_steps = (len(x_test) + batch_size - 1) // batch_size
        print(f"Batch size: {batch_size}")
        if use_stratified:
            # Already printed balanced samples info above, just show summary
            print(f"Training samples available: {len(x_train):,} (using {balanced_samples_per_epoch:,} per epoch)")
        else:
            print(f"Training samples: {len(x_train):,}")
        print(f"Steps per epoch: {steps_per_epoch}")
        
        # Create tf.data.Dataset from generator (NO memory copy!)
        train_dataset = tf.data.Dataset.from_generator(
            train_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, input_size, input_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset - use stratified batches if training is stratified
        if use_stratified:
            print("\nUsing STRATIFIED batch sampling for validation (balanced classes per batch)")
            from StratifiedBatchGenerator import create_stratified_batch_generator
            # Create stratified validation generator (same as training)
            val_generator_func = create_stratified_batch_generator(
                x_test, y_test, batch_size=batch_size, num_classes=10, non_digit_class=None
            )
            # Calculate validation steps for balanced sampling
            # Find minimum class count to match generator's balanced sampling
            from collections import Counter
            class_counts = Counter(y_test)
            min_class_count = min(class_counts.values())
            balanced_samples_per_epoch = min_class_count * 10  # 10 classes
            val_steps = balanced_samples_per_epoch // batch_size
            print(f"  Balanced validation: {min_class_count:,} samples per class = {balanced_samples_per_epoch:,} total")
            print(f"  Validation steps: {val_steps}")
            # Create a single-epoch wrapper for validation (generator must stop after one epoch)
            def val_generator_single_epoch():
                gen = val_generator_func()
                for i in range(val_steps):
                    yield next(gen)
                # Generator stops here (raises StopIteration), allowing Keras to proceed
            # Create tf.data.Dataset from generator
            val_dataset = tf.data.Dataset.from_generator(
                val_generator_single_epoch,
                output_signature=(
                    tf.TensorSpec(shape=(None, input_size, input_size, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.int32)
                )
            )
            # Store val_steps for use in model.fit()
            validation_steps = val_steps
        else:
            # Standard random batching for validation
            val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            val_dataset = val_dataset.batch(batch_size)
            validation_steps = None  # Let Keras calculate automatically
        
        # Callbacks
        softmax_callback = Softmax10DiagnosticsCallback(x_test, y_test, lambda_weight=lambda_weight)
        
        # Choose monitoring metric
        if use_balanced_loss:
            monitor_metric = 'val_loss'  # This will be the BalancedLoss value
            monitor_mode = 'min'  # Minimize loss (lower is better)
            print(f"\nUsing VAL_LOSS (BalancedLoss) metric for monitoring:")
            print(f"  Loss = Mean(Class_Losses) + {lambda_weight} * Variance(Class_Losses)")
            print(f"  Minimizing: Mean class loss + {lambda_weight} * variance")
        else:
            monitor_metric = 'val_loss'  # Standard cross-entropy loss
            monitor_mode = 'min'  # Minimize loss (lower is better)
            print(f"\nUsing VAL_LOSS (cross-entropy) metric for monitoring (default)")
        
        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / "digit_classifier_best.keras"),
                monitor=monitor_metric,
                mode=monitor_mode,
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / "digit_classifier_epoch_{epoch:02d}.keras"),
                save_freq='epoch',
                verbose=0
            ),
            softmax_callback
        ]
        
        # Train with model.fit()
        # validation_steps is set above: val_steps for stratified, None for non-stratified
        history = model.fit(
            train_dataset,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,  # Explicitly set to prevent infinite loop with stratified
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
        
        # Per-class accuracy (softmax mode with 10 classes)
        print("\nPer-class accuracy on test set:")
        print("-" * 40)
        for digit in range(10):
            mask = y_test == digit
            if np.sum(mask) > 0:
                class_accuracy = np.mean(y_pred_classes[mask] == digit)
                correct = np.sum(y_pred_classes[mask] == digit)
                total = np.sum(mask)
                print(f"  Digit {digit}: {class_accuracy:.2%} ({correct}/{total})")
        
        print("="*60)
        print("Digit classifier trained and ready!")
        return model
        
    except Exception as e:
        print(f"Warning: Could not train digit classifier: {e}")
        print("Using untrained model (predictions will be random)")
        return model


def classify_digit(classifier_model, digit_image, input_size=28):
    """
    Classify a single digit image using the CNN model with 10 classes.
    
    Args:
        classifier_model: Trained Keras model (10 classes: digits 0-9 only)
        digit_image: Greyscale image (numpy array), will be resized to input_size x input_size if needed
        input_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (predicted_digit, confidence)
        - predicted_digit: int (0-9)
        - confidence: float (0.0-1.0)
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
    
    # Get predicted class (0-9)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    
    # Return the digit (0-9)
    return predicted_class, confidence, None


def main():
    """
    Standalone training function for the digit classifier.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train a digit classifier on MNIST dataset"
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
    parser.add_argument(
        "--get_USPS",
        action="store_true",
        help="Include USPS dataset in training"
    )
    parser.add_argument(
        "--get_ARDIS",
        action="store_true",
        help="Include ARDIS dataset in training"
    )
    parser.add_argument(
        "--get_CustomONE",
        action="store_true",
        help="Include Custom One dataset in training"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation during training"
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
        

    # Train the model (always uses softmax with 10 classes)
    if args.train_model:
        print(f"Starting digit classifier training with 10 classes (digits 0-9 only)...")
        print(f"Input image size: {input_size}x{input_size}")
        print(f"Datasets: MNIST (base)", end="")
        if args.get_USPS:
            print(" + USPS", end="")
        if args.get_ARDIS:
            print(" + ARDIS", end="")
        if args.get_CustomONE:
            print(" + CustomOne", end="")
        print()
        print(f"Data augmentation: {'ENABLED' if args.augment else 'DISABLED'}")
        print(f"Balanced loss (training): {'ENABLED' if args.balanced_loss else 'DISABLED'}")
        if args.balanced_loss:
            print(f"  Lambda weight: {args.lambda_weight}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Dense layer neurons: {args.dense_layer}")
        
        model = load_or_create_digit_classifier(
            use_stratified=args.stratified,
            num_epochs=args.epoch_count,
            input_size=input_size,
            initial_model_path=args.initial_model,
            use_mnist=True,  # Always use MNIST
            use_usps=args.get_USPS,
            use_ardis=args.get_ARDIS,
            use_custom_one=args.get_CustomONE,
            use_augment=args.augment,
            use_balanced_loss=args.balanced_loss,
            lambda_weight=args.lambda_weight,
            learning_rate=args.learning_rate,
            neurons_in_dense_layer=args.dense_layer
        )
        print("\nTraining complete!")

    print(f"Please specify --train-model to train a new model")
    return 3

if __name__ == "__main__":
    main()
