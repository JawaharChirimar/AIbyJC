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

# Import dataset loaders
from GetArdis import load_ardis_dataset
from GetUSPS import load_usps_dataset

try:
    from emnist import extract_training_samples, extract_test_samples
    EMNIST_AVAILABLE = True
except ImportError:
    EMNIST_AVAILABLE = False
    print("Warning: 'emnist' package not available. Install with: pip install emnist")

# Check numpy version for EMNIST compatibility
try:
    numpy_version = np.__version__
    numpy_major, numpy_minor = map(int, numpy_version.split('.')[:2])
    if numpy_major > 1 or (numpy_major == 1 and numpy_minor >= 25):
        EMNIST_NUMPY_WARNING = True
    else:
        EMNIST_NUMPY_WARNING = False
except:
    EMNIST_NUMPY_WARNING = False

# =============================================================================
# CONFIGURABLE CONSTANTS
# =============================================================================
DROPOUT_RATE = 0.5          # Dropout rate in model (prevents overfitting)
MORPHOLOGY_PROB = 0.5       # Probability of applying morphology augmentation
SIGMOID_THRESHOLD = 0.5     # Threshold for sigmoid classification (output > threshold = detected)
SAMPLE_RATIO = 1.00         # Fraction of data to randomly sample each epoch
AUGMENT_RATIO = 0.25        # Fraction of sampled data to augment
ROTATION_ANGLE = 30         # Maximum rotation angle for augmentation (±degrees)
SHEAR_ANGLE = 15            # Maximum shear angle for augmentation (±degrees)

# Custom non-digit images directory
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"

def load_custom_non_digits():
    """
    Load custom non-digit images from CUSTOM_NON_DIGITS_DIR.
    Also loads individual files like not6.jpeg from the data directory.
    
    Returns:
        numpy array of shape (N, 28, 28, 1), normalized [0,1], or empty array if none found
    """
    custom_images = []
    CUSTOM_NON_DIGITS_DIR = DATA_DIR / "non-digits"
    
    # Load individual not*.jpeg/png files from data directory
    for pattern in ["not*.jpeg", "not*.jpg", "not*.png"]:
        for img_path in DATA_DIR.glob(pattern):
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to 28x28
                    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                    # Normalize to [0, 1]
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    custom_images.append(img_normalized)
                    print(f"  Loaded custom non-digit: {img_path.name}")
            except Exception as e:
                print(f"  Warning: Could not load {img_path}: {e}")
    
    # Load from non_digits directory if it exists
    if CUSTOM_NON_DIGITS_DIR.exists():
        for img_path in CUSTOM_NON_DIGITS_DIR.glob("*"):
            if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png', '.bmp']:
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                        img_normalized = img_resized.astype(np.float32) / 255.0
                        custom_images.append(img_normalized)
                        print(f"  Loaded custom non-digit: {img_path.name}")
                except Exception as e:
                    print(f"  Warning: Could not load {img_path}: {e}")
    
    if custom_images:
        x_custom = np.array(custom_images)
        x_custom = x_custom.reshape(-1, 28, 28, 1)
        return x_custom
    
    return np.array([]).reshape(0, 28, 28, 1)


def create_negative_examples(total_digit_samples, target_ratio=0.133):
    """
    Create negative examples for sigmoid training.
    These are images that are NOT digits, labeled with all zeros.
    
    Args:
        total_digit_samples: Number of digit samples in training set
        target_ratio: Target ratio of negative examples (default ~13.3% = ~40K for 300K digits)
    
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
    n_horiz_cut = int(target_count * 0.05)     # ~5% horizontal cut digits
    n_vert_cut = int(target_count * 0.05)      # ~5% vertical cut digits
    n_pixel_removal = int(target_count * 0.05) # ~5% 80% pixel removal digits
    n_letters = target_count - n_black - n_white - n_sparse_noise - n_dense_noise - n_horiz_cut - n_vert_cut - n_pixel_removal  # ~35% letters
    
    negative_images = []
    
    # 1. All black images
    black_images = np.zeros((n_black, 28, 28, 1), dtype=np.float32)
    negative_images.append(black_images)
    
    # 2. All white images
    white_images = np.ones((n_white, 28, 28, 1), dtype=np.float32)
    negative_images.append(white_images)
    
    # 3. Sparse noise (random scattered dots, 5-15% coverage)
    sparse_noise_images = np.zeros((n_sparse_noise, 28, 28, 1), dtype=np.float32)
    for i in range(n_sparse_noise):
        # Random coverage between 5% and 15%
        coverage = np.random.uniform(0.05, 0.15)
        mask = np.random.random((28, 28)) < coverage
        sparse_noise_images[i, :, :, 0] = mask.astype(np.float32)
    negative_images.append(sparse_noise_images)
    
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
    
    # 5-7. Broken digit images (from EMNIST digits)
    # Load EMNIST digits to create broken versions
    n_broken_total = n_horiz_cut + n_vert_cut + n_pixel_removal
    if n_broken_total > 0:
        try:
            # Use package only - if it fails, skip broken digit negatives
            if EMNIST_AVAILABLE:
                x_digits, _ = extract_training_samples('digits')
                x_digits = x_digits.astype('float32') / 255.0
            else:
                raise Exception("EMNIST package not available")
            
            # 5. Horizontal cut - remove middle third (rows 9-18)
            if n_horiz_cut > 0:
                indices = np.random.choice(len(x_digits), n_horiz_cut, replace=False)
                horiz_cut_images = x_digits[indices].copy()
                horiz_cut_images[:, 9:19, :] = 0  # Black out middle horizontal band
                horiz_cut_images = horiz_cut_images.reshape(-1, 28, 28, 1)
                negative_images.append(horiz_cut_images)
            
            # 6. Vertical cut - remove middle third (cols 9-18)
            if n_vert_cut > 0:
                indices = np.random.choice(len(x_digits), n_vert_cut, replace=False)
                vert_cut_images = x_digits[indices].copy()
                vert_cut_images[:, :, 9:19] = 0  # Black out middle vertical band
                vert_cut_images = vert_cut_images.reshape(-1, 28, 28, 1)
                negative_images.append(vert_cut_images)
            
            # 7. 80% pixel removal - randomly zero out 80% of non-zero pixels
            if n_pixel_removal > 0:
                indices = np.random.choice(len(x_digits), n_pixel_removal, replace=False)
                pixel_removal_images = x_digits[indices].copy()
                for i in range(n_pixel_removal):
                    img = pixel_removal_images[i]
                    # Find non-zero pixels
                    nonzero_mask = img > 0.1
                    nonzero_indices = np.where(nonzero_mask)
                    n_nonzero = len(nonzero_indices[0])
                    if n_nonzero > 0:
                        # Randomly select 80% of non-zero pixels to remove
                        n_to_remove = int(n_nonzero * 0.80)
                        remove_idx = np.random.choice(n_nonzero, n_to_remove, replace=False)
                        for idx in remove_idx:
                            pixel_removal_images[i, nonzero_indices[0][idx], nonzero_indices[1][idx]] = 0
                pixel_removal_images = pixel_removal_images.reshape(-1, 28, 28, 1)
                negative_images.append(pixel_removal_images)
        except Exception as e:
            print(f"  Warning: Could not create broken digit images: {e}")
    
    # 8. Letters from EMNIST Letters dataset
    # Only use letters that CANNOT be confused with digits:
    # A, H, K, M, N, R, V, W, X, Y
    # EMNIST Letters labels: A=1, B=2, ..., Z=26
    SAFE_LETTER_LABELS = [1, 8, 11, 13, 14, 18, 22, 23, 24, 25]  # A, H, K, M, N, R, V, W, X, Y
    
    if EMNIST_AVAILABLE and n_letters > 0:
        try:
            x_letters_all, y_letters_all = extract_training_samples('letters')
            
            # Filter to only safe letters
            safe_mask = np.isin(y_letters_all, SAFE_LETTER_LABELS)
            x_letters = x_letters_all[safe_mask]
            
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
        except Exception as e:
            # Fall back to more noise
            extra_noise = np.zeros((n_letters, 28, 28, 1), dtype=np.float32)
            for i in range(n_letters):
                coverage = np.random.uniform(0.05, 0.15)
                mask = np.random.random((28, 28)) < coverage
                extra_noise[i, :, :, 0] = mask.astype(np.float32)
            negative_images.append(extra_noise)
    else:
        if n_letters > 0:
            extra_noise = np.zeros((n_letters, 28, 28, 1), dtype=np.float32)
            for i in range(n_letters):
                coverage = np.random.uniform(0.05, 0.15)
                mask = np.random.random((28, 28)) < coverage
                extra_noise[i, :, :, 0] = mask.astype(np.float32)
            negative_images.append(extra_noise)
    
    # 9. Load custom non-digit images (e.g., not6.jpeg)
    x_custom = load_custom_non_digits()
    if len(x_custom) > 0:
        negative_images.append(x_custom)
    
    # Combine all negative images
    x_negative = np.concatenate(negative_images, axis=0)
    
    # Labels are all zeros (not any digit)
    y_negative = np.zeros((len(x_negative), 10), dtype=np.float32)
    
    # Shuffle
    indices = np.random.permutation(len(x_negative))
    x_negative = x_negative[indices]
    y_negative = y_negative[indices]
    
    return x_negative, y_negative


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


def create_digit_classifier_model(use_sigmoid=False):
    """
    Create a CNN model for digit classification (0-9).
    
    Uses deep model architecture with 3 conv layers, optimized for EMNIST (240k+ samples).
    
    Args:
        use_sigmoid: Whether to use sigmoid activation instead of softmax (default: False)
    
    Returns:
        Compiled Keras model
    """
    # Model capacity for EMNIST (240k+ samples)
    number_convolution_channels = 32
    number_convolution_channelsF = 64
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
                                    kernel_regularizer=keras.regularizers.l2(0.001))
        accuracy_metric = sigmoid_accuracy
    else:
        output_activation = 'softmax'
        loss_function = 'sparse_categorical_crossentropy'
        output_layer = layers.Dense(10, activation=output_activation)
        accuracy_metric = 'accuracy'
    
    # Deep model architecture (4 conv layers)
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
        layers.Conv2D(number_convolution_channelsF, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(neurons_in_dense_layer, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        output_layer  # 10 classes for digits 0-9
    ])
    
    model.compile(
        optimizer='adam',
        loss=loss_function,
        metrics=[accuracy_metric]
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
    
    Handles digits and non-digits SEPARATELY to ensure consistent ratios per epoch.
    
    Features:
    - sample_ratio: Randomly sample this fraction of data each epoch (default SAMPLE_RATIO)
    - augment_ratio: Augment this fraction of sampled data (default AUGMENT_RATIO)
    """
    def __init__(self, augmentation_pipeline, batch_size=64, sample_ratio=SAMPLE_RATIO, augment_ratio=AUGMENT_RATIO):
        self.augmentation_pipeline = augmentation_pipeline  # Currently unused - kept for potential future use
        self.batch_size = batch_size
        self.sample_ratio = sample_ratio  # Fraction of data to sample each epoch
        self.augment_ratio = augment_ratio  # Fraction of samples to augment
        # Create separate transforms for rotation and shear
        self.rotation_transform = A.Affine(rotate=(-ROTATION_ANGLE, ROTATION_ANGLE), p=1.0)
        self.shear_transform = A.Affine(shear={'x': 0, 'y': (-SHEAR_ANGLE, SHEAR_ANGLE)}, p=1.0)
        # Statistics tracking - separate for digits and negatives
        self.stats = {
            'total_samples': 0,
            'digit_samples': 0,
            'negative_samples': 0,
            'augmented_samples': 0,  # Counts base samples selected for augmentation
            'original_samples': 0,
            'rotation_samples': 0,  # Counts rotation images created
            'shearing_samples': 0,  # Counts shear images created
            'morphology_thicker': 0,
            'morphology_thinner': 0,
            'epoch_digit_count': 0,  # Digit samples used in current epoch
            'epoch_negative_count': 0,  # Negative samples used in current epoch
        }
    
    def _augment_image(self, img, label):
        """
        Process a single image - augment or keep original.
        Returns list of (image, label) tuples (1 for original, 2 for augmented).
        """
        results = []
        
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
            if np.random.random() < MORPHOLOGY_PROB:
                kernel_size = np.random.choice([1, 2])
                if np.random.random() < MORPHOLOGY_PROB:
                    self.stats['morphology_thicker'] += 1
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    img_rotated = cv2.dilate(img_rotated, kernel, iterations=1)
            
            # Convert rotated image back to float [0,1]
            img_rotated_float = (img_rotated.astype(np.float32) / 255.0)
            img_rotated_float = np.expand_dims(img_rotated_float, axis=-1)
            img_rotated_float = np.clip(img_rotated_float, 0.0, 1.0)
            results.append((img_rotated_float, label))
            
            # Create sheared version
            sheared = self.shear_transform(image=img_uint8)
            img_sheared = sheared['image']
            self.stats['shearing_samples'] += 1
            
            # Apply stroke thickness variation to sheared image (optional)
            if np.random.random() < MORPHOLOGY_PROB:
                kernel_size = np.random.choice([1, 2])
                if np.random.random() < MORPHOLOGY_PROB:
                    self.stats['morphology_thicker'] += 1
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    img_sheared = cv2.dilate(img_sheared, kernel, iterations=1)
            
            # Convert sheared image back to float [0,1]
            img_sheared_float = (img_sheared.astype(np.float32) / 255.0)
            img_sheared_float = np.expand_dims(img_sheared_float, axis=-1)
            img_sheared_float = np.clip(img_sheared_float, 0.0, 1.0)
            results.append((img_sheared_float, label))
        else:
            # Keep original image (no augmentation)
            self.stats['original_samples'] += 1
            img_aug_float = (img_uint8.astype(np.float32) / 255.0)
            img_aug_float = np.expand_dims(img_aug_float, axis=-1)
            img_aug_float = np.clip(img_aug_float, 0.0, 1.0)
            results.append((img_aug_float, label))
        
        return results
    
    def flow_separate(self, x_sampled_digits, y_sampled_digits, 
                       x_full_digits, y_full_digits,
                       x_negatives, y_negatives, batch_size=None):
        """
        Generator that yields batches of augmented data.
        
        Handles THREE categories of data:
        1. SAMPLED digits (EMNIST): sample_ratio% sampled each epoch
        2. FULL digits (ARDIS+USPS+Fonts+CustomOne): 100% used every epoch
        3. NEGATIVES: sample_ratio% sampled each epoch
        
        Each epoch:
        - Randomly samples sample_ratio of SAMPLED DIGITS (EMNIST)
        - Uses 100% of FULL DIGITS (ARDIS+USPS+Fonts)
        - Randomly samples sample_ratio of NEGATIVES
        - Combines and shuffles
        - Augments augment_ratio of each
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        num_sampled_digits = len(x_sampled_digits)
        num_full_digits = len(x_full_digits) if x_full_digits is not None and len(x_full_digits) > 0 else 0
        num_negatives = len(x_negatives)
        
        while True:
            # 1. Sample from SAMPLED DIGITS (EMNIST) - sample_ratio%
            sampled_digit_count = int(num_sampled_digits * self.sample_ratio)
            sampled_digit_indices = np.random.choice(num_sampled_digits, sampled_digit_count, replace=False)
            
            # 2. Use ALL of FULL DIGITS (ARDIS+USPS+Fonts) - 100%
            full_digit_count = num_full_digits
            
            # 3. Sample from NEGATIVES - sample_ratio%
            negative_sample_size = int(num_negatives * self.sample_ratio)
            negative_indices = np.random.choice(num_negatives, negative_sample_size, replace=False)
            
            # Total digit count for this epoch
            total_digit_count = sampled_digit_count + full_digit_count
            
            # Update stats
            self.stats['epoch_digit_count'] = total_digit_count
            self.stats['epoch_negative_count'] = negative_sample_size
            
            # Combine samples for this epoch
            arrays_to_concat_x = [x_sampled_digits[sampled_digit_indices]]
            arrays_to_concat_y = [y_sampled_digits[sampled_digit_indices]]
            is_digit_arrays = [np.ones(sampled_digit_count, dtype=bool)]
            
            # Add full digits if available
            if num_full_digits > 0:
                arrays_to_concat_x.append(x_full_digits)
                arrays_to_concat_y.append(y_full_digits)
                is_digit_arrays.append(np.ones(full_digit_count, dtype=bool))
            
            # Add negatives
            arrays_to_concat_x.append(x_negatives[negative_indices])
            arrays_to_concat_y.append(y_negatives[negative_indices])
            is_digit_arrays.append(np.zeros(negative_sample_size, dtype=bool))
            
            epoch_x = np.concatenate(arrays_to_concat_x, axis=0)
            epoch_y = np.concatenate(arrays_to_concat_y, axis=0)
            is_digit = np.concatenate(is_digit_arrays)
            
            # Shuffle together
            shuffle_indices = np.random.permutation(len(epoch_x))
            epoch_x = epoch_x[shuffle_indices]
            epoch_y = epoch_y[shuffle_indices]
            is_digit = is_digit[shuffle_indices]
            
            total_samples = len(epoch_x)
            
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                
                batch_x = epoch_x[start_idx:end_idx]
                batch_y = epoch_y[start_idx:end_idx]
                batch_is_digit = is_digit[start_idx:end_idx]
                
                # Process each image in the batch
                batch_x_aug = []
                batch_y_aug = []
                
                for img, label, is_dig in zip(batch_x, batch_y, batch_is_digit):
                    self.stats['total_samples'] += 1
                    if is_dig:
                        self.stats['digit_samples'] += 1
                    else:
                        self.stats['negative_samples'] += 1
                    
                    # Augment and collect results
                    augmented = self._augment_image(img, label)
                    for aug_img, aug_label in augmented:
                        batch_x_aug.append(aug_img)
                        batch_y_aug.append(aug_label)
                
                batch_x_aug = np.array(batch_x_aug)
                batch_y_aug = np.array(batch_y_aug)
                
                yield batch_x_aug, batch_y_aug
    
    def flow(self, x_sampled, y_sampled, x_full=None, y_full=None, batch_size=None):
        """
        Generator for softmax mode (digits only, no negatives).
        
        Handles TWO categories of data:
        1. SAMPLED digits (EMNIST): sample_ratio% sampled each epoch
        2. FULL digits (ARDIS+USPS+Fonts+CustomOne): 100% used every epoch
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        num_sampled = len(x_sampled)
        num_full = len(x_full) if x_full is not None and len(x_full) > 0 else 0
        
        while True:
            # Sample from SAMPLED data (EMNIST)
            sampled_count = int(num_sampled * self.sample_ratio)
            sampled_indices = np.random.choice(num_sampled, sampled_count, replace=False)
            
            # Use ALL of FULL data (ARDIS+USPS+Fonts)
            full_count = num_full
            
            total_count = sampled_count + full_count
            self.stats['epoch_digit_count'] = total_count
            
            # Combine samples for this epoch
            if num_full > 0:
                epoch_x = np.concatenate([x_sampled[sampled_indices], x_full], axis=0)
                epoch_y = np.concatenate([y_sampled[sampled_indices], y_full], axis=0)
            else:
                epoch_x = x_sampled[sampled_indices]
                epoch_y = y_sampled[sampled_indices]
            
            # Shuffle
            shuffle_indices = np.random.permutation(len(epoch_x))
            epoch_x = epoch_x[shuffle_indices]
            epoch_y = epoch_y[shuffle_indices]
            
            for start_idx in range(0, total_count, batch_size):
                end_idx = min(start_idx + batch_size, total_count)
                
                batch_x = epoch_x[start_idx:end_idx]
                batch_y = epoch_y[start_idx:end_idx]
                
                batch_x_aug = []
                batch_y_aug = []
                
                for img, label in zip(batch_x, batch_y):
                    self.stats['total_samples'] += 1
                    self.stats['digit_samples'] += 1
                    
                    augmented = self._augment_image(img, label)
                    for aug_img, aug_label in augmented:
                        batch_x_aug.append(aug_img)
                        batch_y_aug.append(aug_label)
                
                batch_x_aug = np.array(batch_x_aug)
                batch_y_aug = np.array(batch_y_aug)
                
                yield batch_x_aug, batch_y_aug


class AugmentationStatsCallback(keras.callbacks.Callback):
    """
    Custom callback to print per-epoch augmentation statistics.
    Tracks how many samples were augmented vs original each epoch.
    Shows separate digit/negative counts when using flow_separate.
    """
    def __init__(self, datagen, samples_per_epoch):
        self.datagen = datagen
        self.samples_per_epoch = samples_per_epoch
        self.last_total = 0
        self.last_augmented = 0
        self.last_original = 0
        self.last_rotation = 0
        self.last_shearing = 0
        self.last_digits = 0
        self.last_negatives = 0
    
    def on_epoch_end(self, epoch, logs=None):
        stats = self.datagen.stats
        current_total = stats['total_samples']
        current_augmented = stats['augmented_samples']
        current_original = stats['original_samples']
        current_rotation = stats['rotation_samples']
        current_shearing = stats['shearing_samples']
        current_digits = stats['digit_samples']
        current_negatives = stats['negative_samples']
        
        samples_this_epoch = current_total - self.last_total
        augmented_this_epoch = current_augmented - self.last_augmented
        original_this_epoch = current_original - self.last_original
        rotation_this_epoch = current_rotation - self.last_rotation
        shearing_this_epoch = current_shearing - self.last_shearing
        digits_this_epoch = current_digits - self.last_digits
        negatives_this_epoch = current_negatives - self.last_negatives
        
        self.last_total = current_total
        self.last_augmented = current_augmented
        self.last_original = current_original
        self.last_rotation = current_rotation
        self.last_shearing = current_shearing
        self.last_digits = current_digits
        self.last_negatives = current_negatives
        
        # Show separate digit/negative counts if using flow_separate
        if negatives_this_epoch > 0:
            print(f"\n[Epoch {epoch+1}] Base samples: {samples_this_epoch:,} (Digits: {digits_this_epoch:,}, Non-digits: {negatives_this_epoch:,})")
        else:
            print(f"\n[Epoch {epoch+1}] Base samples processed: {samples_this_epoch:,} (Augmented: {augmented_this_epoch:,}, Original: {original_this_epoch:,})")
        print(f"  Images created: Rotation: {rotation_this_epoch:,}, Shearing: {shearing_this_epoch:,}, Original: {original_this_epoch:,}")


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


def load_font_digits(use_sigmoid=False):
    """
    Load font-generated digit images if available.
    
    Args:
        use_sigmoid: Whether to load sigmoid (one-hot) or softmax (integer) labels
    
    Returns:
        Tuple of (x_data, y_data) or (None, None) if not available
    """
    
    if use_sigmoid:
        npz_path = DATA_DIR / "font_digits" / "font_digits_sigmoid.npz"
    else:
        npz_path = DATA_DIR / "font_digits" / "font_digits_softmax.npz"
    
    if npz_path.exists():
        try:
            data = np.load(npz_path)
            x_data = data['x']
            y_data = data['y']
            print(f"Loaded font digits: {len(x_data)} samples from {npz_path}")
            return x_data, y_data
        except Exception as e:
            print(f"Warning: Could not load font digits from {npz_path}: {e}")
            return None, None
    else:
        print(f"Font digits not found at {npz_path}")
        print("Generate with: python FontDigitGenerator.py --api-key YOUR_KEY --num-fonts 100")
        return None, None


def load_custom_one(use_sigmoid=False):
    """
    Load custom "1" digit variations (serif style with no base).
    
    Args:
        use_sigmoid: Whether to load sigmoid (one-hot) or softmax (integer) labels
    
    Returns:
        Tuple of (x_data, y_data) or (None, None) if not available
    """
    
    if use_sigmoid:
        npz_path = DATA_DIR / "custom_one" / "custom_one_sigmoid.npz"
    else:
        npz_path = DATA_DIR / "custom_one" / "custom_one_softmax.npz"
    
    if npz_path.exists():
        try:
            data = np.load(npz_path)
            x_data = data['x']
            y_data = data['y']
            print(f"Loaded custom '1' digits: {len(x_data)} samples from {npz_path}")
            return x_data, y_data
        except Exception as e:
            print(f"Warning: Could not load custom '1' from {npz_path}: {e}")
            return None, None
    else:
        print(f"Custom '1' digits not found at {npz_path}")
        print("Generate with: python GenerateCustomOne.py")
        return None, None


def load_and_combine_datasets(use_sigmoid=False):
    """
    Load and combine all digit datasets:
    - EMNIST Digits (sampled 70% per epoch) - includes MNIST
    - ARDIS (used 100% every epoch)
    - USPS (used 100% every epoch)
    - Font Digits (used 100% every epoch)
    - Custom "1" digits (used 100% every epoch) - serif style with no base
    
    Note: MNIST is NOT loaded separately since EMNIST Digits already includes it.
    
    Args:
        use_sigmoid: Whether to prepare labels for sigmoid (affects font digit loading)
    
    Returns:
        Tuple of (x_sampled_train, y_sampled_train, x_full_train, y_full_train, x_test, y_test)
        - x_sampled_train/y_sampled_train: EMNIST (will be 70% sampled per epoch)
        - x_full_train/y_full_train: ARDIS + USPS + Fonts + CustomOne (used 100% every epoch)
        - x_test/y_test: Combined test set from all datasets
        Arrays are normalized to [0, 1] and reshaped to (samples, 28, 28, 1)
    """
    # Datasets that will be sampled 70% per epoch
    sampled_datasets = []
    sampled_names = []
    
    # Datasets that will be used 100% every epoch
    full_datasets = []
    full_names = []
    
    # Test datasets (always combined)
    test_datasets = []
    
    # =========================================================================
    # SAMPLED DATASETS (70% per epoch): EMNIST Digits (includes MNIST)
    # =========================================================================
    
    # Load EMNIST Digits (includes MNIST - no need to load MNIST separately)
    print("Loading EMNIST Digits dataset (includes MNIST)...")
    
    if EMNIST_AVAILABLE:
        try:
            x_train_emnist, y_train_emnist = extract_training_samples('digits')
            x_test_emnist, y_test_emnist = extract_test_samples('digits')
            
            # Convert to numpy arrays if needed
            x_train_emnist = np.asarray(x_train_emnist, dtype=np.uint8)
            y_train_emnist = np.asarray(y_train_emnist, dtype=np.uint8)
            x_test_emnist = np.asarray(x_test_emnist, dtype=np.uint8)
            y_test_emnist = np.asarray(y_test_emnist, dtype=np.uint8)
            
            train_len = len(x_train_emnist)
            test_len = len(x_test_emnist)
            print(f"  EMNIST Digits: {train_len} training, {test_len} test samples")
            sampled_datasets.append((x_train_emnist, y_train_emnist))
            sampled_names.append(f"EMNIST ({train_len})")
            test_datasets.append((x_test_emnist, y_test_emnist))
        except Exception as e:
            error_msg = str(e)
            print(f"  Warning: Could not load EMNIST Digits: {error_msg}")
            print("  Falling back to MNIST only...")
            # Fallback to MNIST
            print("Loading MNIST dataset...")
            (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()
            print(f"  MNIST: {len(x_train_mnist)} training, {len(x_test_mnist)} test samples")
            sampled_datasets.append((x_train_mnist, y_train_mnist))
            sampled_names.append(f"MNIST ({len(x_train_mnist)})")
            test_datasets.append((x_test_mnist, y_test_mnist))
    else:
        print("  EMNIST package not available.")
        print("  Falling back to MNIST only...")
        # Fallback to MNIST if EMNIST not available
        print("Loading MNIST dataset...")
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()
        print(f"  MNIST: {len(x_train_mnist)} training, {len(x_test_mnist)} test samples")
        sampled_datasets.append((x_train_mnist, y_train_mnist))
        sampled_names.append(f"MNIST ({len(x_train_mnist)})")
        test_datasets.append((x_test_mnist, y_test_mnist))
    
    # =========================================================================
    # FULL DATASETS (100% every epoch): ARDIS, USPS, Font Digits
    # =========================================================================
    
    # Load ARDIS
    ardis_data = load_ardis_dataset()
    if ardis_data[0] is not None:
        x_train_ardis, y_train_ardis, x_test_ardis, y_test_ardis = ardis_data
        full_datasets.append((x_train_ardis, y_train_ardis))
        full_names.append(f"ARDIS ({len(x_train_ardis)})")
        test_datasets.append((x_test_ardis, y_test_ardis))
    
    # Load USPS
    usps_data = load_usps_dataset()
    if usps_data[0] is not None:
        x_train_usps, y_train_usps, x_test_usps, y_test_usps = usps_data
        full_datasets.append((x_train_usps, y_train_usps))
        full_names.append(f"USPS ({len(x_train_usps)})")
        test_datasets.append((x_test_usps, y_test_usps))
    
    # Load font-generated digits if available
    print("\nChecking for font-generated digits...")
    x_fonts, y_fonts = load_font_digits(use_sigmoid=False)  # Always load integer labels first
    if x_fonts is not None:
        # Split font digits: 80% train, 20% test
        n_fonts = len(x_fonts)
        n_train = int(n_fonts * 0.8)
        indices = np.random.permutation(n_fonts)
        
        x_fonts_train = x_fonts[indices[:n_train]]
        y_fonts_train = y_fonts[indices[:n_train]]
        x_fonts_test = x_fonts[indices[n_train:]]
        y_fonts_test = y_fonts[indices[n_train:]]
        
        # Font data is already normalized and shaped correctly (N, 28, 28, 1)
        # Convert back to uint8 for consistency
        x_fonts_train_flat = (x_fonts_train.squeeze(-1) * 255).astype(np.uint8)
        x_fonts_test_flat = (x_fonts_test.squeeze(-1) * 255).astype(np.uint8)
        
        full_datasets.append((x_fonts_train_flat, y_fonts_train))
        full_names.append(f"Fonts ({n_train})")
        test_datasets.append((x_fonts_test_flat, y_fonts_test))
    
    # FULL DATASET 4: Custom "1" digits (serif style with no base) - 100% every epoch
    x_custom_one, y_custom_one = load_custom_one(use_sigmoid=False)
    if x_custom_one is not None:
        # Split custom one: 80% train, 20% test
        n_custom = len(x_custom_one)
        n_train = int(n_custom * 0.8)
        indices = np.random.permutation(n_custom)
        
        x_custom_train = x_custom_one[indices[:n_train]]
        y_custom_train = y_custom_one[indices[:n_train]]
        x_custom_test = x_custom_one[indices[n_train:]]
        y_custom_test = y_custom_one[indices[n_train:]]
        
        # Custom one data is already normalized and shaped correctly (N, 28, 28, 1)
        # Convert back to uint8 for consistency
        x_custom_train_flat = (x_custom_train.squeeze(-1) * 255).astype(np.uint8)
        x_custom_test_flat = (x_custom_test.squeeze(-1) * 255).astype(np.uint8)
        
        full_datasets.append((x_custom_train_flat, y_custom_train))
        full_names.append(f"CustomOne ({n_train})")
        test_datasets.append((x_custom_test_flat, y_custom_test))
    
    # =========================================================================
    # COMBINE DATASETS
    # =========================================================================
    
    # Combine sampled datasets (EMNIST)
    if len(sampled_datasets) > 0:
        x_sampled_train = np.concatenate([ds[0] for ds in sampled_datasets], axis=0)
        y_sampled_train = np.concatenate([ds[1] for ds in sampled_datasets], axis=0)
    else:
        x_sampled_train = np.array([]).reshape(0, 28, 28)
        y_sampled_train = np.array([])
    
    # Combine full datasets (ARDIS + USPS + Fonts + CustomOne)
    if len(full_datasets) > 0:
        x_full_train = np.concatenate([ds[0] for ds in full_datasets], axis=0)
        y_full_train = np.concatenate([ds[1] for ds in full_datasets], axis=0)
    else:
        x_full_train = np.array([]).reshape(0, 28, 28)
        y_full_train = np.array([])
    
    # Combine all test datasets
    x_test = np.concatenate([ds[0] for ds in test_datasets], axis=0)
    y_test = np.concatenate([ds[1] for ds in test_datasets], axis=0)
    
    # Print summary
    print(f"\n=== Dataset Summary ===")
    print(f"SAMPLED datasets (70% per epoch): {' + '.join(sampled_names)}")
    print(f"  Total sampled training: {len(x_sampled_train)}")
    print(f"FULL datasets (100% every epoch): {' + '.join(full_names) if full_names else 'None'}")
    print(f"  Total full training: {len(x_full_train)}")
    print(f"Combined test set: {len(x_test)} samples")
    print(f"========================\n")
    
    # Normalize pixel values to [0, 1]
    x_sampled_train = x_sampled_train.astype('float32') / 255.0
    x_full_train = x_full_train.astype('float32') / 255.0 if len(x_full_train) > 0 else x_full_train.astype('float32')
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN (28, 28, 1)
    x_sampled_train = x_sampled_train.reshape(x_sampled_train.shape[0], 28, 28, 1)
    if len(x_full_train) > 0:
        x_full_train = x_full_train.reshape(x_full_train.shape[0], 28, 28, 1)
    else:
        x_full_train = x_full_train.reshape(0, 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    return x_sampled_train, y_sampled_train, x_full_train, y_full_train, x_test, y_test


def load_or_create_digit_classifier(classifier_model_path=None, 
train_model=True, num_epochs=20, use_sigmoid=False):
    """
    Load a pre-trained digit classifier or create/train a new one.
    
    Always uses EMNIST (if available), data augmentation, and 4 layer CNN model.
    
    Args:
        classifier_model_path: Path to saved classifier model (.keras file)
        train_model: Whether to train a new model (True) or load existing (False)
        num_epochs: Number of training epochs (default: 20)
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
            # Provide custom objects for loading (including custom metrics)
            custom_objects = {'sigmoid_accuracy': sigmoid_accuracy}
            model = keras.models.load_model(classifier_model_path, custom_objects=custom_objects)
            print("Digit classifier loaded successfully")
            return model
        except Exception as e:
            print(f"Warning: Could not load classifier from {classifier_model_path}: {e}")
            print("Creating new classifier model...")
    
    # We're going to train a new model, so create the run directory now
    # Create timestamped directory for model checkpoints
    base_dir = DATA_DIR / "modelForDE"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory: run_YYYY_MM_DD_HH_MM_SS
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model checkpoints will be saved to: {run_dir}")
    
    # Create new model
    print("Creating new digit classifier model...")
    model = create_digit_classifier_model(use_sigmoid=use_sigmoid)
    
    # Try to train on all digit datasets
    try:
        # Load and combine datasets
        # - sampled: EMNIST (will be 70% sampled per epoch)
        # - full: ARDIS + USPS + Fonts + CustomOne (used 100% every epoch)
        x_sampled_train, y_sampled_train, x_full_train, y_full_train, x_test, y_test = \
            load_and_combine_datasets(use_sigmoid=use_sigmoid)
        
        # Store for digit tracking
        x_digits_test = x_test.copy()
        y_digits_test = y_test.copy()
        
        # Total digit counts for reference
        total_sampled_digits = len(x_sampled_train)  # EMNIST (will be 70% sampled)
        total_full_digits = len(x_full_train)         # ARDIS + USPS + Fonts (100% used)
        total_digits = total_sampled_digits + total_full_digits
        
        print(f"\nDigit data breakdown:")
        print(f"  SAMPLED (70% per epoch): {total_sampled_digits:,} (EMNIST)")
        print(f"  FULL (100% every epoch): {total_full_digits:,} (ARDIS + USPS + Fonts + CustomOne)")
        print(f"  Total digits: {total_digits:,}")
        
        # For sigmoid mode, create negative examples and keep them SEPARATE
        x_negative_train = None
        y_negative_train = None
        x_negative_test = None
        y_negative_test = None
        
        # Convert labels to one-hot if using sigmoid activation
        if use_sigmoid:
            print("\nConverting labels to one-hot encoding for binary_crossentropy...")
            y_sampled_train = keras.utils.to_categorical(y_sampled_train, 10)
            if len(y_full_train) > 0:
                y_full_train = keras.utils.to_categorical(y_full_train, 10)
            y_digits_test = keras.utils.to_categorical(y_digits_test, 10)
            
            # Create negative examples (non-digits) - keep SEPARATE from digits
            print("\nCreating negative examples for sigmoid training...")
            print(f"  Target ratio: 13.3% of digit samples")
            x_negative_train, y_negative_train = create_negative_examples(total_digits, target_ratio=0.133)
            x_negative_test, y_negative_test = create_negative_examples(len(x_digits_test), target_ratio=0.133)
            
            print(f"  Total negative examples - Train: {len(x_negative_train):,}, Test: {len(x_negative_test):,}")
            print(f"\nTraining data:")
            print(f"  Digits: {total_digits:,} ({total_sampled_digits:,} sampled + {total_full_digits:,} full)")
            print(f"  Non-digits: {len(x_negative_train):,}")
            print(f"Test data: {len(x_digits_test):,} digits + {len(x_negative_test):,} non-digits")
            
            # For validation, we still need combined test data
            x_test = np.concatenate([x_digits_test, x_negative_test], axis=0)
            y_test = np.concatenate([y_digits_test, y_negative_test], axis=0)
            indices = np.random.permutation(len(x_test))
            x_test = x_test[indices]
            y_test = y_test[indices]
        
        print(f"\nTest samples: {len(x_test)}")
        print(f"Number of epochs: {num_epochs}")
        
        # Create data augmentation pipeline
        print("\nSetting up data augmentation pipeline...")
        augmentation_pipeline = create_augmentation_pipeline()
        
        # Create data generator with augmentation (uses class defaults)
        train_datagen = ImageDataGeneratorWithAugmentation(
            augmentation_pipeline=augmentation_pipeline
        )
        
        # Train the model with augmented data
        print("Starting training with data augmentation...")
        print("\n" + "="*60)
        print("Augmentation Configuration:")
        print("="*60)
        
        # Use generator's ratios (single source of truth)
        sample_ratio = train_datagen.sample_ratio
        augment_ratio = train_datagen.augment_ratio
        
        # Calculate samples per epoch:
        # - SAMPLED datasets (EMNIST): 70% sampled each epoch
        # - FULL datasets (ARDIS+USPS+Fonts+CustomOne): 100% used every epoch
        sampled_per_epoch = int(total_sampled_digits * sample_ratio)
        full_per_epoch = total_full_digits  # 100% of ARDIS, USPS, Fonts
        digits_per_epoch = sampled_per_epoch + full_per_epoch
        
        print(f"Digit sampling per epoch:")
        print(f"  SAMPLED (EMNIST): {sample_ratio*100:.0f}% of {total_sampled_digits:,} = {sampled_per_epoch:,}")
        print(f"  FULL (ARDIS+USPS+Fonts+CustomOne): 100% of {total_full_digits:,} = {full_per_epoch:,}")
        print(f"  Total digits per epoch: {digits_per_epoch:,}")
        
        if use_sigmoid:
            # Separate sampling for negatives too
            neg_samples_per_epoch = int(len(x_negative_train) * sample_ratio)
            total_samples_per_epoch = digits_per_epoch + neg_samples_per_epoch
            print(f"  Non-digits: {sample_ratio*100:.0f}% of {len(x_negative_train):,} = {neg_samples_per_epoch:,}")
            print(f"  Total samples per epoch: {total_samples_per_epoch:,}")
        else:
            total_samples_per_epoch = digits_per_epoch
        
        print(f"\nAugmentation: {augment_ratio*100:.0f}% of sampled data will be augmented")
        print(f"              {(1-augment_ratio)*100:.0f}% will remain original")
        print("Each augmented sample produces 2 images: one rotated, one sheared")
        print("\nAugmentation details:")
        print("  Rotation AND Shearing: 100% of augmented samples (each produces 2 images)")
        print(f"    - Rotation: ±{ROTATION_ANGLE}° rotation (no shift, no scale)")
        print(f"    - Shearing: ±{SHEAR_ANGLE}° vertical shear (no shift, no scale)")
        print("  Morphology - Stroke dilation: ~25% of augmented images (50% chance × 50% dilation)")
        print(f"\nNote: Different random {sample_ratio*100:.0f}% sample each epoch for better generalization")
        augmented_per_epoch = int(total_samples_per_epoch * augment_ratio)
        original_per_epoch = total_samples_per_epoch - augmented_per_epoch
        total_images_per_epoch = augmented_per_epoch * 2 + original_per_epoch  # augmented*2 + original
        print(f"Images per epoch: ~{total_images_per_epoch:,}")
        print(f"  (~{augmented_per_epoch * 2:,} from augmented [each producing 2], ~{original_per_epoch:,} original)")
        print(f"Total over {num_epochs} epochs: ~{total_images_per_epoch * num_epochs:,}")
        print("="*60 + "\n")
        
        stats_callback = AugmentationStatsCallback(train_datagen, total_samples_per_epoch)
        
        # ModelCheckpoint callback to save model after each epoch with epoch number
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
        
        # steps_per_epoch based on total sampled data
        steps_per_epoch = total_samples_per_epoch // train_datagen.batch_size
        
        # Use flow_separate for sigmoid mode (separate digit/negative handling)
        # Use flow for softmax mode (digits only)
        if use_sigmoid:
            data_generator = train_datagen.flow_separate(
                x_sampled_train, y_sampled_train,  # EMNIST (70% sampled)
                x_full_train, y_full_train,        # ARDIS+USPS+Fonts+CustomOne (100% used)
                x_negative_train, y_negative_train  # Negatives (70% sampled)
            )
        else:
            data_generator = train_datagen.flow(
                x_sampled_train, y_sampled_train,  # EMNIST (70% sampled)
                x_full_train, y_full_train         # ARDIS+USPS+Fonts (100% used)
            )
        
        model.fit(
            data_generator,
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
            
            # Show digit vs negative breakdown if using sigmoid
            if use_sigmoid and stats['digit_samples'] > 0:
                print(f"\nDigit vs Non-digit breakdown:")
                print(f"  Digits processed: {stats['digit_samples']:,} ({stats['digit_samples']/total*100:.1f}%)")
                print(f"  Non-digits processed: {stats['negative_samples']:,} ({stats['negative_samples']/total*100:.1f}%)")
                print(f"  Last epoch: {stats['epoch_digit_count']:,} digits + {stats['epoch_negative_count']:,} non-digits")
            
            total_images = stats['rotation_samples'] + stats['shearing_samples'] + stats['original_samples']
            print(f"\nTotal images created across all epochs: {total_images}")
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
                
                # Also check if highest output > threshold for digits
                digit_max_outputs = np.max(y_pred[digit_mask], axis=1)
                digit_confident = np.mean(digit_max_outputs > SIGMOID_THRESHOLD)
                print(f"Digits with max output > {SIGMOID_THRESHOLD}: {digit_confident*100:.1f}%")
            
            # Negative rejection rate (ALL outputs should be < threshold)
            if n_negatives > 0:
                neg_preds = y_pred[negative_mask]
                neg_max_outputs = np.max(neg_preds, axis=1)
                neg_correct = np.sum(neg_max_outputs < SIGMOID_THRESHOLD)
                neg_acc = neg_correct / n_negatives
                print(f"\nNegative rejection rate (all outputs < {SIGMOID_THRESHOLD}): {neg_acc*100:.1f}% ({neg_correct}/{n_negatives})")
                
                # Show distribution of max outputs for negatives
                neg_failed = neg_max_outputs >= SIGMOID_THRESHOLD
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
        "--epoch-count",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--use-sigmoid",
        action="store_true",
        help="Use sigmoid activation instead of softmax (softmax is used by default)"
    )
    
    args = parser.parse_args()
        
    # Train the model
    print("Starting digit classifier training...")
    model = load_or_create_digit_classifier(
        classifier_model_path=args.model_path, 
        train_model=args.train_model,
        num_epochs=args.epoch_count,
        use_sigmoid=args.use_sigmoid
    )
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

