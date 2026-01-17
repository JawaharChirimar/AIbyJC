#!/usr/bin/env python3
"""
DigitClassifier.py

Provides functions for creating, training, and using a CNN-based digit classifier.
The classifier is trained on the MNIST and EMNIST Digits datasets to recognize handwritten digits (0-9).
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

try:
    from emnist import extract_training_samples, extract_test_samples
    EMNIST_AVAILABLE = True
except ImportError:
    EMNIST_AVAILABLE = False
    print("Warning: 'emnist' package not available. Install with: pip install emnist")


def create_negative_examples(total_digit_samples, target_ratio=0.10):
    """
    Create negative examples for sigmoid training.
    These are images that are NOT digits, labeled with all zeros.
    
    Args:
        total_digit_samples: Number of digit samples in training set
        target_ratio: Target ratio of negative examples (default 10%)
    
    Returns:
        Tuple of (x_negative, y_negative) where:
        - x_negative: numpy array of shape (n, 28, 28, 1), normalized [0,1]
        - y_negative: numpy array of shape (n, 10), all zeros
    """
    target_count = int(total_digit_samples * target_ratio)
    
    # Allocate counts for each type
    n_black = target_count // 10           # ~10% all black
    n_white = target_count // 10           # ~10% all white  
    n_sparse_noise = int(target_count * 0.15)  # ~15% sparse noise (5-15% coverage)
    n_dense_noise = int(target_count * 0.15)   # ~15% dense uniform noise (50-80% coverage)
    n_letters = target_count - n_black - n_white - n_sparse_noise - n_dense_noise  # ~50% letters
    
    print(f"\nCreating negative examples (~{target_ratio*100:.0f}% of data):")
    print(f"  All black: {n_black}")
    print(f"  All white: {n_white}")
    print(f"  Sparse noise: {n_sparse_noise}")
    print(f"  Dense noise: {n_dense_noise}")
    print(f"  Letters: {n_letters}")
    
    negative_images = []
    
    # 1. All black images
    black_images = np.zeros((n_black, 28, 28, 1), dtype=np.float32)
    negative_images.append(black_images)
    print(f"  Generated {n_black} all-black images")
    
    # 2. All white images
    white_images = np.ones((n_white, 28, 28, 1), dtype=np.float32)
    negative_images.append(white_images)
    print(f"  Generated {n_white} all-white images")
    
    # 3. Sparse noise (random scattered dots, 5-15% coverage)
    sparse_noise_images = np.zeros((n_sparse_noise, 28, 28, 1), dtype=np.float32)
    for i in range(n_sparse_noise):
        # Random coverage between 5% and 15%
        coverage = np.random.uniform(0.05, 0.15)
        mask = np.random.random((28, 28)) < coverage
        sparse_noise_images[i, :, :, 0] = mask.astype(np.float32)
    negative_images.append(sparse_noise_images)
    print(f"  Generated {n_sparse_noise} sparse noise images")
    
    # 4. Dense uniform noise (truly random, 50-80% coverage)
    # Each pixel is independently random - avoids accidental digit patterns
    dense_noise_images = np.zeros((n_dense_noise, 28, 28, 1), dtype=np.float32)
    for i in range(n_dense_noise):
        # Random coverage between 50% and 80%
        coverage = np.random.uniform(0.50, 0.80)
        # Uniform random - every pixel independently sampled
        mask = np.random.random((28, 28)) < coverage
        dense_noise_images[i, :, :, 0] = mask.astype(np.float32)
    negative_images.append(dense_noise_images)
    print(f"  Generated {n_dense_noise} dense uniform noise images")
    
    # 5. Letters from EMNIST Letters dataset
    # Only use letters that CANNOT be confused with digits:
    # A, H, K, M, N, R, V, W, X, Y
    # EMNIST Letters labels: A=1, B=2, ..., Z=26
    SAFE_LETTER_LABELS = [1, 8, 11, 13, 14, 18, 22, 23, 24, 25]  # A, H, K, M, N, R, V, W, X, Y
    
    if EMNIST_AVAILABLE and n_letters > 0:
        try:
            print(f"  Loading EMNIST Letters dataset...")
            x_letters_all, y_letters_all = extract_training_samples('letters')
            
            # Filter to only safe letters
            safe_mask = np.isin(y_letters_all, SAFE_LETTER_LABELS)
            x_letters = x_letters_all[safe_mask]
            print(f"  Filtered to safe letters (A,H,K,M,N,R,V,W,X,Y): {len(x_letters)} samples")
            
            # Normalize and reshape
            x_letters = x_letters.astype('float32') / 255.0
            x_letters = x_letters.reshape(-1, 28, 28, 1)
            
            # Randomly sample n_letters from the dataset
            if len(x_letters) > n_letters:
                indices = np.random.choice(len(x_letters), n_letters, replace=False)
                x_letters = x_letters[indices]
            else:
                # If not enough letters, use all and repeat if needed
                if len(x_letters) < n_letters:
                    repeats = (n_letters // len(x_letters)) + 1
                    x_letters = np.tile(x_letters, (repeats, 1, 1, 1))[:n_letters]
            
            negative_images.append(x_letters)
            print(f"  Sampled {len(x_letters)} letter images from EMNIST Letters")
        except Exception as e:
            print(f"  Warning: Could not load EMNIST Letters: {e}")
            print(f"  Generating additional noise images instead...")
            # Fall back to more noise
            extra_noise = np.zeros((n_letters, 28, 28, 1), dtype=np.float32)
            for i in range(n_letters):
                coverage = np.random.uniform(0.05, 0.15)
                mask = np.random.random((28, 28)) < coverage
                extra_noise[i, :, :, 0] = mask.astype(np.float32)
            negative_images.append(extra_noise)
    else:
        if n_letters > 0:
            print(f"  EMNIST not available, generating additional noise images...")
            extra_noise = np.zeros((n_letters, 28, 28, 1), dtype=np.float32)
            for i in range(n_letters):
                coverage = np.random.uniform(0.05, 0.15)
                mask = np.random.random((28, 28)) < coverage
                extra_noise[i, :, :, 0] = mask.astype(np.float32)
            negative_images.append(extra_noise)
    
    # Combine all negative images
    x_negative = np.concatenate(negative_images, axis=0)
    
    # Labels are all zeros (not any digit)
    y_negative = np.zeros((len(x_negative), 10), dtype=np.float32)
    
    # Shuffle
    indices = np.random.permutation(len(x_negative))
    x_negative = x_negative[indices]
    y_negative = y_negative[indices]
    
    print(f"  Total negative examples: {len(x_negative)}")
    
    return x_negative, y_negative


def create_digit_classifier_model(use_240k_samples=False, use_deep_model=True, use_sigmoid=False):
    """
    Create a CNN model for digit classification (0-9).
    
    Args:
        use_240k_samples: Whether to use 240k samples from EMNIST Digits (affects model capacity)
        use_deep_model: Whether to use deep model architecture (default: True)
        use_sigmoid: Whether to use sigmoid activation instead of softmax (default: False)
    
    Returns:
        Compiled Keras model
    """

    # Adjust model capacity based on dataset size
    if use_240k_samples:
        # Larger model for MNIST + EMNIST (more training data)
        number_convolution_channels = 32
        neurons_in_dense_layer = 64
    else:
        # Smaller model for MNIST only
        number_convolution_channels = 64
        neurons_in_dense_layer = 128
    
    # Output activation and loss function based on use_sigmoid flag
    if use_sigmoid:
        output_activation = 'sigmoid'
        loss_function = 'binary_crossentropy'
        # Initialize biases negative so default output is LOW (~5%)
        # Forces model to learn "see features → increase output" 
        # instead of "default high → suppress on other features"
        output_layer = layers.Dense(10, activation=output_activation,
                                    bias_initializer=keras.initializers.Constant(-3.0),
                                    bias_regularizer=keras.regularizers.l2(0.01))
    else:
        output_activation = 'softmax'
        loss_function = 'sparse_categorical_crossentropy'
        output_layer = layers.Dense(10, activation=output_activation)
        
    if use_deep_model:
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(neurons_in_dense_layer, activation='elu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            output_layer  # 10 classes for digits 0-9
        ])
    else:
        # Shallow model architecture (fewer layers)
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(neurons_in_dense_layer, activation='elu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            output_layer  # 10 classes for digits 0-9
        ])
    
    model.compile(
        optimizer='adam',
        loss=loss_function,
        metrics=['accuracy']
    )
    
    return model


def create_augmentation_pipeline(stats_tracker=None):
    """
    Create an albumentations augmentation pipeline for MNIST/EMNIST digit training.
    
    Note: This pipeline is currently NOT used - kept for potential future use.
    Rotation/shear and morphology are applied directly in ImageDataGeneratorWithAugmentation.
    
    Args:
        stats_tracker: Optional dict to track augmentation statistics
    
    Returns:
        albumentations Compose object (currently unused)
    """
    # Note: Rotation and Slant (shear) are applied in ImageDataGeneratorWithAugmentation
    # Each augmented sample produces 2 images: one rotated, one sheared
    # Blur/noise below are NOT currently applied (pipeline is unused)
    
    transforms = []
    transforms.append(A.GaussianBlur(blur_limit=(1, 3), p=0.3))  # Light blur
    transforms.append(A.GaussNoise(p=0.2))  # Light noise
    
    transform = A.Compose(transforms, p=1.0)
    
    return transform


class ImageDataGeneratorWithAugmentation:
    """
    Custom data generator that applies albumentations augmentations to data.
    For augmented samples: each produces 2 images (one rotated, one sheared).
    
    Features:
    - sample_ratio: Randomly sample this fraction of data each epoch (default 0.70)
    - augment_ratio: Augment this fraction of sampled data (default 0.70)
    """
    def __init__(self, augmentation_pipeline, batch_size=64, sample_ratio=0.70, augment_ratio=0.70):
        self.augmentation_pipeline = augmentation_pipeline  # Currently unused - kept for potential future use
        self.batch_size = batch_size
        self.sample_ratio = sample_ratio  # Fraction of data to sample each epoch
        self.augment_ratio = augment_ratio  # Fraction of samples to augment
        # Create separate transforms for rotation and shear
        self.rotation_transform = A.Affine(rotate=(-48, 48), p=1.0)
        self.shear_transform = A.Affine(shear={'x': 0, 'y': (-15, 15)}, p=1.0)
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'augmented_samples': 0,  # Counts base samples selected for augmentation
            'original_samples': 0,
            'rotation_samples': 0,  # Counts rotation images created
            'shearing_samples': 0,  # Counts shear images created
            'morphology_thicker': 0,
            'morphology_thinner': 0,
            'epoch_sample_count': 0,  # Samples used in current epoch
        }
    
    def flow(self, x, y, batch_size=None):
        """
        Generator that yields batches of augmented data.
        
        Each epoch:
        - Randomly samples sample_ratio (default 70%) of data
        - Augments augment_ratio (default 70%) of those samples
        - Different random sample each epoch for better generalization
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        num_samples = len(x)
        
        while True:
            # At the start of each epoch, randomly sample data
            # This creates variety across epochs - different subset each time
            sample_size = int(num_samples * self.sample_ratio)
            epoch_indices = np.random.choice(num_samples, sample_size, replace=False)
            np.random.shuffle(epoch_indices)
            self.stats['epoch_sample_count'] = sample_size
            
            for start_idx in range(0, sample_size, batch_size):
                end_idx = min(start_idx + batch_size, sample_size)
                batch_indices = epoch_indices[start_idx:end_idx]
                
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]
                
                # Process each image in the batch
                batch_x_aug = []
                batch_y_aug = []
                
                for img, label in zip(batch_x, batch_y):
                    self.stats['total_samples'] += 1
                    
                    # Convert from (28, 28, 1) to (28, 28) for albumentations
                    img_2d = img.squeeze(axis=-1)
                    
                    # Convert from float [0,1] to uint8 [0,255] for albumentations
                    img_uint8 = (img_2d * 255).astype(np.uint8)
                    
                    # Apply augmentation to augment_ratio of samples
                    # Each augmented sample produces 2 images: one rotated, one sheared
                    if np.random.random() < self.augment_ratio:
                        self.stats['augmented_samples'] += 1
                        
                        # Create rotated version
                        rotated = self.rotation_transform(image=img_uint8)
                        img_rotated = rotated['image']
                        self.stats['rotation_samples'] += 1
                        
                        # Apply stroke thickness variation to rotated image (optional)
                        if np.random.random() < 0.5:
                            kernel_size = np.random.choice([1, 2])
                            if np.random.random() < 0.5:
                                self.stats['morphology_thicker'] += 1
                                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                                img_rotated = cv2.dilate(img_rotated, kernel, iterations=1)
                        
                        # Convert rotated image back to float [0,1]
                        img_rotated_float = (img_rotated.astype(np.float32) / 255.0)
                        img_rotated_float = np.expand_dims(img_rotated_float, axis=-1)
                        img_rotated_float = np.clip(img_rotated_float, 0.0, 1.0)
                        batch_x_aug.append(img_rotated_float)
                        batch_y_aug.append(label)
                        
                        # Create sheared version
                        sheared = self.shear_transform(image=img_uint8)
                        img_sheared = sheared['image']
                        self.stats['shearing_samples'] += 1
                        
                        # Apply stroke thickness variation to sheared image (optional)
                        if np.random.random() < 0.5:
                            kernel_size = np.random.choice([1, 2])
                            if np.random.random() < 0.5:
                                self.stats['morphology_thicker'] += 1
                                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                                img_sheared = cv2.dilate(img_sheared, kernel, iterations=1)
                        
                        # Convert sheared image back to float [0,1]
                        img_sheared_float = (img_sheared.astype(np.float32) / 255.0)
                        img_sheared_float = np.expand_dims(img_sheared_float, axis=-1)
                        img_sheared_float = np.clip(img_sheared_float, 0.0, 1.0)
                        batch_x_aug.append(img_sheared_float)
                        batch_y_aug.append(label)
                    else:
                        # Keep original image (no augmentation)
                        self.stats['original_samples'] += 1
                        img_aug = img_uint8
                        
                        # Convert back to float [0,1]
                        img_aug_float = (img_aug.astype(np.float32) / 255.0)
                        img_aug_float = np.expand_dims(img_aug_float, axis=-1)
                        img_aug_float = np.clip(img_aug_float, 0.0, 1.0)
                        batch_x_aug.append(img_aug_float)
                        batch_y_aug.append(label)
                
                batch_x_aug = np.array(batch_x_aug)
                batch_y_aug = np.array(batch_y_aug)
                
                yield batch_x_aug, batch_y_aug


class AugmentationStatsCallback(keras.callbacks.Callback):
    """
    Custom callback to print per-epoch augmentation statistics.
    Tracks how many samples were augmented vs original each epoch.
    """
    def __init__(self, datagen, samples_per_epoch):
        self.datagen = datagen
        self.samples_per_epoch = samples_per_epoch
        self.last_total = 0
        self.last_augmented = 0
        self.last_original = 0
        self.last_rotation = 0
        self.last_shearing = 0
    
    def on_epoch_end(self, epoch, logs=None):
        stats = self.datagen.stats
        current_total = stats['total_samples']
        current_augmented = stats['augmented_samples']
        current_original = stats['original_samples']
        current_rotation = stats['rotation_samples']
        current_shearing = stats['shearing_samples']
        
        samples_this_epoch = current_total - self.last_total
        augmented_this_epoch = current_augmented - self.last_augmented
        original_this_epoch = current_original - self.last_original
        rotation_this_epoch = current_rotation - self.last_rotation
        shearing_this_epoch = current_shearing - self.last_shearing
        
        self.last_total = current_total
        self.last_augmented = current_augmented
        self.last_original = current_original
        self.last_rotation = current_rotation
        self.last_shearing = current_shearing
        
        print(f"\n[Epoch {epoch+1}] Base samples processed: {samples_this_epoch} (Augmented: {augmented_this_epoch}, Original: {original_this_epoch})")
        print(f"  Images created: Rotation: {rotation_this_epoch}, Shearing: {shearing_this_epoch}, Original: {original_this_epoch}")


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
            results.append(f"Digits: {digit_acc:.1f}%")
        
        # Negative rejection rate
        if self.n_negatives > 0:
            neg_max = np.max(y_pred[self.negative_mask], axis=1)
            neg_rejected = np.sum(neg_max < 0.5)
            neg_acc = neg_rejected / self.n_negatives * 100
            results.append(f"Neg rejected: {neg_acc:.1f}% ({neg_rejected}/{self.n_negatives})")
        
        print(f"  [Sigmoid] {' | '.join(results)}")


def load_and_combine_datasets(use_mnist=True, use_emnist=True):
    """
    Load and combine MNIST and/or EMNIST datasets.
    
    Args:
        use_mnist: Whether to load MNIST dataset (default: True)
        use_emnist: Whether to load EMNIST Digits dataset (default: True)
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) as numpy arrays
        Arrays are normalized to [0, 1] and reshaped to (samples, 28, 28, 1)
    
    Raises:
        ValueError: If no datasets could be loaded
    """
    # Load MNIST if requested
    x_train_mnist = None
    y_train_mnist = None
    x_test_mnist = None
    y_test_mnist = None
    
    if use_mnist:
        print("Loading MNIST dataset...")
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()
        print(f"Loaded MNIST: {len(x_train_mnist)} training, {len(x_test_mnist)} test samples")
    else:
        print("Skipping MNIST dataset (use_mnist=False)")
    
    # Load EMNIST Digits if requested and available
    x_train_emnist = None
    y_train_emnist = None
    x_test_emnist = None
    y_test_emnist = None
    
    if use_emnist and EMNIST_AVAILABLE:
        try:
            print("Loading EMNIST Digits dataset...")
            # Note: 'digits' parameter loads ONLY EMNIST Digits (0-9), not the full EMNIST dataset
            # This gives us ~240,000 additional digit samples beyond MNIST's 60,000
            x_train_emnist, y_train_emnist = extract_training_samples('digits')
            x_test_emnist, y_test_emnist = extract_test_samples('digits')
            print(f"Loaded EMNIST Digits: {len(x_train_emnist)} training, {len(x_test_emnist)} test samples")
            if len(x_train_emnist) == 0:
                print("Warning: EMNIST Digits loaded but has 0 samples. Check dataset installation.")
                x_train_emnist = None
        except Exception as e:
            print(f"Error: Could not load EMNIST Digits: {e}")
            import traceback
            traceback.print_exc()
            x_train_emnist = None
            y_train_emnist = None
            x_test_emnist = None
            y_test_emnist = None
    elif use_emnist and not EMNIST_AVAILABLE:
        print("Skipping EMNIST dataset (EMNIST package not available. Install with: pip install emnist)")
    elif not use_emnist:
        print("Skipping EMNIST dataset (use_emnist=False)")
    
    # Combine datasets based on what's available
    datasets_to_combine = []
    dataset_names = []
    
    if x_train_mnist is not None:
        datasets_to_combine.append((x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist))
        dataset_names.append(f"MNIST ({len(x_train_mnist)} samples)")
    
    if x_train_emnist is not None:
        datasets_to_combine.append((x_train_emnist, y_train_emnist, x_test_emnist, y_test_emnist))
        dataset_names.append(f"EMNIST Digits ({len(x_train_emnist)} samples)")
    
    # Check if we have at least one dataset
    if len(datasets_to_combine) == 0:
        error_msg = "No training data available! "
        if not use_mnist and not use_emnist:
            error_msg += "Both MNIST and EMNIST are disabled."
        elif not use_mnist:
            error_msg += "MNIST is disabled (use_mnist=False) and EMNIST is not available or failed to load."
        elif not use_emnist:
            error_msg += "EMNIST is disabled (use_emnist=False) and MNIST failed to load."
        else:
            error_msg += "Both datasets failed to load."
        raise ValueError(error_msg)
    
    # Combine all available datasets
    if len(datasets_to_combine) == 1:
        x_train, y_train, x_test, y_test = datasets_to_combine[0]
        print(f"Using {dataset_names[0]}: {len(x_train)} training, {len(x_test)} test samples")
    else:
        # Combine multiple datasets
        print(f"Combining datasets: {' + '.join(dataset_names)}")
        x_train = np.concatenate([ds[0] for ds in datasets_to_combine], axis=0)
        y_train = np.concatenate([ds[1] for ds in datasets_to_combine], axis=0)
        x_test = np.concatenate([ds[2] for ds in datasets_to_combine], axis=0)
        y_test = np.concatenate([ds[3] for ds in datasets_to_combine], axis=0)
        print(f"Combined dataset: {len(x_train)} training, {len(x_test)} test samples")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    return x_train, y_train, x_test, y_test


def load_or_create_digit_classifier(classifier_model_path=None, 
train_model=True,
use_augmentation=True, use_mnist=True, use_emnist=True, 
num_epochs=20, use_deep_model=True, use_sigmoid=False):
    """
    Load a pre-trained digit classifier or create/train a new one.
    
    Args:
        classifier_model_path: Path to saved classifier model (.keras file)
        use_augmentation: Whether to use data augmentation during training (default: True)
        use_mnist: Whether to include MNIST data for training/validation (default: True)
        use_emnist: Whether to include EMNIST Digits data for training/validation (default: True)
        num_epochs: Number of training epochs (default: 20)
        use_deep_model: Whether to use deep model architecture (default: True)
        use_sigmoid: Whether to use sigmoid activation instead of softmax (default: False)
    
    Returns:
        Trained Keras model for digit classification
    """

    print("===========train_model: ", train_model)
    print("===========classifier_model_path: ", classifier_model_path)
    # Try to load existing model (from specified path only if train_model is False)
    if (not train_model) and classifier_model_path and os.path.exists(classifier_model_path):
        try:
            print(f"Loading digit classifier from: {classifier_model_path}")
            model = keras.models.load_model(classifier_model_path)
            print("Digit classifier loaded successfully")
            return model
        except Exception as e:
            print(f"Warning: Could not load classifier from {classifier_model_path}: {e}")
            print("Creating new classifier model...")
    
    # We're going to train a new model, so create the run directory now
    # Create timestamped directory for model checkpoints
    base_dir = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data" / "modelForDE"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory: run_YYYY_MM_DD_HH_MM_SS
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model checkpoints will be saved to: {run_dir}")
    
    # Create new model
    print("Creating new digit classifier model...")
    model = create_digit_classifier_model(use_240k_samples=use_emnist, 
    use_deep_model=use_deep_model, use_sigmoid=use_sigmoid)
    
    # Try to train on MNIST + EMNIST Digits dataset
    try:
        # Load and combine datasets
        x_train, y_train, x_test, y_test = load_and_combine_datasets(use_mnist=use_mnist, use_emnist=use_emnist)
        
        # Convert labels to one-hot if using sigmoid activation
        if use_sigmoid:
            print("Converting labels to one-hot encoding for binary_crossentropy...")
            y_train = keras.utils.to_categorical(y_train, 10)
            y_test = keras.utils.to_categorical(y_test, 10)
            
            # Add negative examples (non-digits) for sigmoid training
            print("\nAdding negative examples for sigmoid training...")
            x_negative_train, y_negative_train = create_negative_examples(len(x_train), target_ratio=0.10)
            
            # Combine with training data
            x_train = np.concatenate([x_train, x_negative_train], axis=0)
            y_train = np.concatenate([y_train, y_negative_train], axis=0)
            
            # Shuffle the combined training data
            indices = np.random.permutation(len(x_train))
            x_train = x_train[indices]
            y_train = y_train[indices]
            
            print(f"Combined training data: {len(x_train)} samples (including ~10% negative examples)")
            
            # Also add negative examples to test set
            print("\nAdding negative examples to test set...")
            x_negative_test, y_negative_test = create_negative_examples(len(x_test), target_ratio=0.10)
            
            x_test = np.concatenate([x_test, x_negative_test], axis=0)
            y_test = np.concatenate([y_test, y_negative_test], axis=0)
            
            # Shuffle the combined test data
            indices = np.random.permutation(len(x_test))
            x_test = x_test[indices]
            y_test = y_test[indices]
            
            print(f"Combined test data: {len(x_test)} samples (including ~10% negative examples)")
        
        print(f"Training samples per epoch: {len(x_train)}")
        print(f"Test samples: {len(x_test)}")
        print(f"Number of epochs: {num_epochs}")
        
        # Setup training based on whether augmentation is enabled
        if use_augmentation:
            # Create data augmentation pipeline
            # This will be applied on-the-fly during training
            print("Setting up data augmentation pipeline...")
            augmentation_pipeline = create_augmentation_pipeline()
            
            # Create data generator with augmentation
            # sample_ratio=0.70: Use 70% of data each epoch (different random subset)
            # augment_ratio=0.70: Augment 70% of sampled data (each produces 2 images)
            train_datagen = ImageDataGeneratorWithAugmentation(
                augmentation_pipeline=augmentation_pipeline,
                batch_size=64,
                sample_ratio=0.70,
                augment_ratio=0.70
            )
            
            # Train the model with augmented data
            print("Starting training with data augmentation...")
            print("\n" + "="*60)
            print("Augmentation Configuration:")
            print("="*60)
            sample_ratio = 0.70
            augment_ratio = 0.70
            samples_per_epoch = int(len(x_train) * sample_ratio)
            print(f"Data sampling: {sample_ratio*100:.0f}% of data randomly sampled each epoch ({samples_per_epoch:,} samples)")
            print(f"Augmentation: {augment_ratio*100:.0f}% of sampled data will be augmented")
            print(f"              {(1-augment_ratio)*100:.0f}% will remain original")
            print("Each augmented sample produces 2 images: one rotated, one sheared")
            print("\nAugmentation details:")
            print("  Rotation AND Shearing: 100% of augmented samples (each produces 2 images)")
            print("    - Rotation: ±48° rotation (no shift, no scale)")
            print("    - Shearing: ±15° vertical shear (no shift, no scale)")
            print("  Morphology - Stroke dilation: ~25% of augmented images (50% chance × 50% dilation)")
            print("\nNote: Different random 70% sample each epoch for better generalization")
            augmented_per_epoch = int(samples_per_epoch * augment_ratio)
            original_per_epoch = samples_per_epoch - augmented_per_epoch
            total_images_per_epoch = augmented_per_epoch * 2 + original_per_epoch  # augmented*2 + original
            print(f"Images per epoch: ~{total_images_per_epoch:,}")
            print(f"  (~{augmented_per_epoch * 2:,} from augmented [each producing 2], ~{original_per_epoch:,} original)")
            print(f"Total over {num_epochs} epochs: ~{total_images_per_epoch * num_epochs:,}")
            print("="*60 + "\n")
            
            stats_callback = AugmentationStatsCallback(train_datagen, len(x_train))
            
            # ModelCheckpoint callback to save model after each epoch with epoch number
            # Save all epoch models in the timestamped run directory
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / "digit_classifier_epoch_{epoch:02d}.keras"),
                save_best_only=False,  # Save every epoch, not just best
                save_weights_only=False,  # Save full model
                verbose=0  # Don't print save messages (already verbose=1 in fit)
            )
            
            print(f"Epoch models will be saved as: {run_dir}/digit_classifier_epoch_XX.keras (one per epoch)")
            
            # Build callbacks list
            callbacks_list = [stats_callback, checkpoint_callback]
            
            # Add sigmoid diagnostics callback if using sigmoid
            if use_sigmoid:
                sigmoid_callback = SigmoidDiagnosticsCallback(x_test, y_test)
                callbacks_list.append(sigmoid_callback)
                print("Sigmoid diagnostics will be printed after each epoch")
            
            # steps_per_epoch based on 70% sampled data
            # Note: actual images per step varies due to augmentation (some samples produce 2 images)
            steps_per_epoch = int(len(x_train) * sample_ratio) // 64
            
            model.fit(
                train_datagen.flow(x_train, y_train, batch_size=64),
                steps_per_epoch=steps_per_epoch,
                epochs=num_epochs,
                validation_data=(x_test, y_test),
                verbose=1,
                callbacks=callbacks_list
            )
            
            # Print final augmentation statistics
            print("\n" + "="*60)
            print("Final Augmentation Statistics:")
            print("="*60)
            stats = train_datagen.stats
            total = stats['total_samples']
            if total > 0:
                print(f"Total base samples processed across all epochs: {total}")
                print(f"Average base samples per epoch: {total / num_epochs:.0f}")
                total_images = stats['rotation_samples'] + stats['shearing_samples'] + stats['original_samples']
                print(f"Total images created across all epochs: {total_images}")
                print(f"Average images per epoch: {total_images / num_epochs:.0f}")
                print(f"\nBase sample breakdown:")
                print(f"  Augmented base samples: {stats['augmented_samples']}/{total} ({stats['augmented_samples']/total*100:.1f}%)")
                print(f"  Original base samples: {stats['original_samples']}/{total} ({stats['original_samples']/total*100:.1f}%)")
                print(f"\nImage creation breakdown:")
                print(f"  Rotation images: {stats['rotation_samples']}/{total_images} ({stats['rotation_samples']/total_images*100:.1f}% of images)")
                print(f"  Shearing images: {stats['shearing_samples']}/{total_images} ({stats['shearing_samples']/total_images*100:.1f}% of images)")
                print(f"  Original images: {stats['original_samples']}/{total_images} ({stats['original_samples']/total_images*100:.1f}% of images)")
                print(f"\nMorphology augmentation application:")
                if stats['augmented_samples'] > 0:
                    total_aug_images = stats['rotation_samples'] + stats['shearing_samples']
                    print(f"  Morphology - Thicker strokes: {stats['morphology_thicker']}/{total_aug_images} ({stats['morphology_thicker']/total_aug_images*100:.1f}% of augmented images)")
                print(f"\nNote: Each augmented base sample produces 2 images (one rotated, one sheared).")
                print(f"      Morphology stats are tracked manually.")
            print("="*60)
        else:
            # Train the model without augmentation
            print("Starting training WITHOUT data augmentation...")
            print("\n" + "="*60)
            print("Training Configuration:")
            print("="*60)
            print("Training with original data only (no augmentation)")
            print(f"Training samples per epoch: {len(x_train)}")
            print(f"Test samples: {len(x_test)}")
            print("="*60 + "\n")
            
            # ModelCheckpoint callback to save model after each epoch with epoch number
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / "digit_classifier_epoch_{epoch:02d}.keras"),
                save_best_only=False,  # Save every epoch, not just best
                save_weights_only=False,  # Save full model
                verbose=0  # Don't print save messages (already verbose=1 in fit)
            )
            
            print(f"Epoch models will be saved as: {run_dir}/digit_classifier_epoch_XX.keras (one per epoch)")
            
            model.fit(
                x_train, y_train,
                batch_size=64,
                epochs=num_epochs,
                validation_data=(x_test, y_test),
                verbose=1,
                callbacks=[checkpoint_callback]
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
        
        # Sigmoid mode: detailed diagnostic for digits vs negatives
        if use_sigmoid:
            print("\n" + "-"*40)
            print("SIGMOID MODE DIAGNOSTICS:")
            print("-"*40)
            
            # Identify digits vs negatives
            # Digits have a "1" somewhere in label, negatives are all zeros
            digit_mask = np.sum(y_test, axis=1) > 0
            negative_mask = ~digit_mask
            
            n_digits = np.sum(digit_mask)
            n_negatives = np.sum(negative_mask)
            print(f"Test set composition: {n_digits} digits, {n_negatives} negatives")
            
            # Digit classification accuracy (argmax matches)
            if n_digits > 0:
                digit_preds = np.argmax(y_pred[digit_mask], axis=1)
                digit_labels = np.argmax(y_test[digit_mask], axis=1)
                digit_acc = np.mean(digit_preds == digit_labels)
                digit_correct = np.sum(digit_preds == digit_labels)
                print(f"\nDigit classification accuracy: {digit_acc*100:.1f}% ({digit_correct}/{n_digits})")
                
                # Also check if highest output > 0.5 for digits
                digit_max_outputs = np.max(y_pred[digit_mask], axis=1)
                digit_confident = np.mean(digit_max_outputs > 0.5)
                print(f"Digits with max output > 0.5: {digit_confident*100:.1f}%")
            
            # Negative rejection rate (ALL outputs should be < 0.5)
            if n_negatives > 0:
                neg_preds = y_pred[negative_mask]
                neg_max_outputs = np.max(neg_preds, axis=1)
                neg_correct = np.sum(neg_max_outputs < 0.5)
                neg_acc = neg_correct / n_negatives
                print(f"\nNegative rejection rate (all outputs < 0.5): {neg_acc*100:.1f}% ({neg_correct}/{n_negatives})")
                
                # Show distribution of max outputs for negatives
                neg_failed = neg_max_outputs >= 0.5
                if np.sum(neg_failed) > 0:
                    failed_max = neg_max_outputs[neg_failed]
                    failed_classes = np.argmax(neg_preds[neg_failed], axis=1)
                    print(f"Failed negatives: {np.sum(neg_failed)}")
                    print(f"  Max output range: {failed_max.min():.3f} - {failed_max.max():.3f}")
                    print(f"  Most common false class: {np.bincount(failed_classes).argmax()}")
            
            print("-"*40)
            
            # Per-class accuracy for digits only (one-hot labels)
            y_test_classes = np.argmax(y_test[digit_mask], axis=1)
            y_pred_classes = np.argmax(y_pred[digit_mask], axis=1)
            
            print("\nPer-class accuracy on DIGITS only:")
            print("-" * 40)
            for digit in range(10):
                mask = y_test_classes == digit
                if np.sum(mask) > 0:
                    class_accuracy = np.mean(y_pred_classes[mask] == digit)
                    correct = np.sum(y_pred_classes[mask] == digit)
                    total = np.sum(mask)
                    print(f"  Digit {digit}: {class_accuracy:.2%} ({correct}/{total})")
        else:
            # Softmax mode: standard per-class accuracy
            y_pred_classes = np.argmax(y_pred, axis=1)
            
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


def classify_digit(classifier_model, digit_image):
    """
    Classify a single digit image using the CNN model.
    
    Args:
        classifier_model: Trained Keras model
        digit_image: 28x28 greyscale image (numpy array)
    
    Returns:
        Predicted digit (0-9) and confidence score
    """
    # Ensure image is the right shape and type
    if digit_image.shape != (28, 28):
        # Resize if needed
        digit_image = cv2.resize(digit_image, (28, 28))
    
    # Normalize pixel values to [0, 1]
    digit_normalized = digit_image.astype('float32') / 255.0
    
    # The input image should already be in MNIST format: white digits on black background
    # (ensured by BoundingBoxFromYolo.py preprocessing)
    # MNIST: white digits (high values ~1.0) on black background (low values ~0.0)
    
    # Reshape for model input: (1, 28, 28, 1)
    digit_input = digit_normalized.reshape(1, 28, 28, 1)
    
    # Predict
    predictions = classifier_model.predict(digit_input, verbose=0)
    print("predictions: ", predictions)
    predicted_digit = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_digit])
    
    return predicted_digit, confidence


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
        "--no-augment",
        action="store_true",
        help="Disable data augmentation during training (augmentation is enabled by default)"
    )
    parser.add_argument(
        "--no-mnist",
        action="store_true",
        help="Exclude MNIST data from training/validation (MNIST is included by default)"
    )
    parser.add_argument(
        "--no-emnist",
        action="store_true",
        help="Exclude EMNIST Digits data from training/validation (EMNIST is included by default)"
    )
    parser.add_argument(
        "--epoch-count",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--no-deep-model",
        action="store_true",
        help="Use shallow model architecture instead of deep model (deep model is used by default)"
    )
    parser.add_argument(
        "--use-sigmoid",
        action="store_true",
        help="Use sigmoid activation instead of softmax (softmax is used by default)"
    )
    
    args = parser.parse_args()
        
    # Train the model
    print("Starting digit classifier training...")
    use_augmentation = not args.no_augment  # Augmentation is default (True unless --no-augment is set)
    use_mnist = not args.no_mnist  # MNIST is default (True unless --no-mnist is set)
    use_emnist = not args.no_emnist  # EMNIST is default (True unless --no-emnist is set)
    use_deep_model = not args.no_deep_model  # Deep model is default (True unless --no-deep-model is set)
    use_sigmoid = args.use_sigmoid  # Sigmoid is opt-in (False unless --use-sigmoid is set)
    model = load_or_create_digit_classifier(
        args.model_path, 
        args.train_model,
        use_augmentation=use_augmentation,
        use_mnist=use_mnist,
        use_emnist=use_emnist,
        num_epochs=args.epoch_count,
        use_deep_model=use_deep_model,
        use_sigmoid=use_sigmoid
    )
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

