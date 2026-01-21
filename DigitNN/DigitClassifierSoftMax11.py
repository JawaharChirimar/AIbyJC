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
SIGMOID_THRESHOLD = 0.5  # Threshold for sigmoid classification (output > threshold = detected)
SAMPLE_RATIO = 1.00         # Fraction of data to randomly sample each epoch
AUGMENT_RATIO = 0.25        # Fraction of sampled data to augment
ROTATION_ANGLE = 30         # Maximum rotation angle for augmentation (±degrees)
SHEAR_ANGLE = 15            # Maximum shear angle for augmentation (±degrees)
NEGATIVE_RATIO = 0.20       # Ratio of negative examples to digit samples (20% = ~56K for 282K digits)

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


def create_negative_examples(total_digit_samples, target_ratio=NEGATIVE_RATIO):
    """
    Create negative examples for softmax training with 11 classes.
    These are images that are NOT digits, labeled as class 10.
    
    Implements comprehensive negative generation according to detailed spec:
    - Easy (20%): All Black, All White, Sparse/Dense Noise, Salt & Pepper, Gradient
    - Medium (35%): Letters (Safe + Additional), Symbols, Cuts, Pixel Removal, Custom
    - Hard (40%): Partials, Multiple Cuts, Random Erasures, Blurred, Faint, Rotated, 
                   Distorted, Overlapping, Stroke Breaks, Hard Mining
    
    Args:
        total_digit_samples: Number of digit samples in training set
        target_ratio: Target ratio of negative examples (default NEGATIVE_RATIO = 20%)
    
    Returns:
        Tuple of (x_negative, y_negative) where:
        - x_negative: numpy array of shape (n, 28, 28, 1), normalized [0,1]
        - y_negative: numpy array of shape (n,), all labeled as 10 (not a digit)
    """
    target_count = int(total_digit_samples * target_ratio)
    
    # Allocate counts according to spec (percentages are relative to total negative count)
    # Easy (20%)
    n_black = int(target_count * 0.03)           # 3% = 1,270
    n_white = int(target_count * 0.03)           # 3% = 1,270
    n_sparse_noise = int(target_count * 0.05)    # 5% = 2,115
    n_dense_noise = int(target_count * 0.04)     # 4% = 1,690
    n_salt_pepper = int(target_count * 0.02)     # 2% = 850
    n_gradient = int(target_count * 0.03)         # 3% = 1,270
    
    # Medium (35%)
    n_letters_safe = int(target_count * 0.20)    # 20% = 8,460
    n_letters_additional = int(target_count * 0.05)  # 5% = 2,115
    n_symbols = int(target_count * 0.03)         # 3% = 1,270
    n_horiz_cut = int(target_count * 0.02)       # 2% = 850
    n_vert_cut = int(target_count * 0.02)        # 2% = 850
    n_pixel_removal = int(target_count * 0.02)    # 2% = 850
    n_custom = int(target_count * 0.01)          # 1% = 420
    
    # Hard (40%)
    n_partial_top = int(target_count * 0.03)     # 3% = 1,270
    n_partial_bottom = int(target_count * 0.03)  # 3% = 1,270
    n_partial_left = int(target_count * 0.02)    # 2% = 850
    n_partial_right = int(target_count * 0.02)   # 2% = 850
    n_multiple_cuts = int(target_count * 0.02)   # 2% = 850
    n_random_erasures = int(target_count * 0.03) # 3% = 1,270
    n_blurred = int(target_count * 0.03)         # 3% = 1,270
    n_faint = int(target_count * 0.02)           # 2% = 850
    n_rotated_90 = int(target_count * 0.02)       # 2% = 850
    n_rotated_180 = int(target_count * 0.02)      # 2% = 850
    n_distorted = int(target_count * 0.02)        # 2% = 850
    n_overlapping = int(target_count * 0.02)       # 2% = 850
    n_stroke_breaks = int(target_count * 0.02)    # 2% = 850
    n_hard_mining = int(target_count * 0.05)      # 5% = 2,115 (placeholder - requires iterative training)
    
    negative_images = []
    
    print(f"\nCreating {target_count:,} negative examples ({target_ratio*100:.0f}% of {total_digit_samples:,} digits)...")
    
    # ============================================================================
    # EASY NEGATIVES (20% = 8,460)
    # ============================================================================
    
    # 1. All black images (3%)
    if n_black > 0:
        black_images = np.zeros((n_black, 28, 28, 1), dtype=np.float32)
        negative_images.append(black_images)
        print(f"  All Black: {n_black:,}")
    
    # 2. All white images (3%)
    if n_white > 0:
        white_images = np.ones((n_white, 28, 28, 1), dtype=np.float32)
        negative_images.append(white_images)
        print(f"  All White: {n_white:,}")
    
    # 3. Sparse noise (5%)
    if n_sparse_noise > 0:
        sparse_noise_images = np.zeros((n_sparse_noise, 28, 28, 1), dtype=np.float32)
        for i in range(n_sparse_noise):
            coverage = np.random.uniform(0.05, 0.15)
            mask = np.random.random((28, 28)) < coverage
            sparse_noise_images[i, :, :, 0] = mask.astype(np.float32)
        negative_images.append(sparse_noise_images)
        print(f"  Sparse Noise: {n_sparse_noise:,}")
    
    # 4. Dense noise (4%)
    if n_dense_noise > 0:
        dense_noise_images = np.zeros((n_dense_noise, 28, 28, 1), dtype=np.float32)
        for i in range(n_dense_noise):
            coverage = np.random.uniform(0.50, 0.80)
            mask = np.random.random((28, 28)) < coverage
            dense_noise_images[i, :, :, 0] = mask.astype(np.float32)
        negative_images.append(dense_noise_images)
        print(f"  Dense Noise: {n_dense_noise:,}")
    
    # 5. Salt & Pepper noise (2%)
    if n_salt_pepper > 0:
        salt_pepper_images = np.zeros((n_salt_pepper, 28, 28, 1), dtype=np.float32)
        for i in range(n_salt_pepper):
            img = np.zeros((28, 28))
            salt_pepper_mask = np.random.random((28, 28)) < 0.1  # 10% pixels
            img[salt_pepper_mask] = np.random.choice([0.0, 1.0], size=np.sum(salt_pepper_mask))
            salt_pepper_images[i, :, :, 0] = img
        negative_images.append(salt_pepper_images)
        print(f"  Salt & Pepper: {n_salt_pepper:,}")
    
    # 6. Gradient noise (3%)
    if n_gradient > 0:
        gradient_images = np.zeros((n_gradient, 28, 28, 1), dtype=np.float32)
        for i in range(n_gradient):
            x = np.linspace(0, 1, 28)
            y = np.linspace(0, 1, 28)
            X, Y = np.meshgrid(x, y)
            gradient = (X + Y) / 2
            noise = np.random.random((28, 28)) * 0.3
            img = np.clip(gradient + noise, 0, 1)
            gradient_images[i, :, :, 0] = img
        negative_images.append(gradient_images)
        print(f"  Gradient Noise: {n_gradient:,}")
    
    # ============================================================================
    # MEDIUM NEGATIVES (35% = 14,805)
    # ============================================================================
    
    # Load EMNIST digits for broken/partial/rotated negatives
    x_digits = None
    y_digits = None
    n_digit_based_needed = (n_horiz_cut + n_vert_cut + n_pixel_removal + 
                           n_partial_top + n_partial_bottom + n_partial_left + n_partial_right +
                           n_multiple_cuts + n_random_erasures + n_blurred + n_faint +
                           n_rotated_90 + n_rotated_180 + n_distorted + n_overlapping + n_stroke_breaks)
    
    if n_digit_based_needed > 0:
        try:
            if EMNIST_AVAILABLE:
                x_digits, y_digits = extract_training_samples('digits')
                x_digits = x_digits.astype('float32') / 255.0
                # Reshape if needed
                if len(x_digits.shape) == 3:
                    x_digits = x_digits.reshape(-1, 28, 28, 1)
            else:
                raise Exception("EMNIST package not available")
        except Exception as e:
            print(f"  Warning: Could not load EMNIST digits for broken/partial negatives: {e}")
            x_digits = None
    
    # 7. Letters - Safe (20%)
    SAFE_LETTER_LABELS = [1, 8, 11, 13, 14, 18, 22, 23, 24, 25]  # A, H, K, M, N, R, V, W, X, Y
    if EMNIST_AVAILABLE and n_letters_safe > 0:
        try:
            x_letters_all, y_letters_all = extract_training_samples('letters')
            safe_mask = np.isin(y_letters_all, SAFE_LETTER_LABELS)
            x_letters = x_letters_all[safe_mask]
            x_letters = x_letters.astype('float32') / 255.0
            x_letters = x_letters.reshape(-1, 28, 28, 1)
            
            if len(x_letters) > n_letters_safe:
                indices = np.random.choice(len(x_letters), n_letters_safe, replace=False)
                x_letters = x_letters[indices]
            else:
                if len(x_letters) < n_letters_safe:
                    repeats = (n_letters_safe // len(x_letters)) + 1
                    x_letters = np.tile(x_letters, (repeats, 1, 1, 1))[:n_letters_safe]
            
            negative_images.append(x_letters)
            print(f"  Letters (Safe): {len(x_letters):,}")
        except Exception as e:
            print(f"  Warning: Could not load safe letters: {e}")
    
    # 8. Letters - Additional (5%) - B, C, D, E, F, G, J, P, Q, T, U (exclude O, I, S, Z, L)
    ADDITIONAL_LETTER_LABELS = [2, 3, 4, 5, 6, 7, 10, 16, 17, 20, 21]  # B, C, D, E, F, G, J, P, Q, T, U
    if EMNIST_AVAILABLE and n_letters_additional > 0:
        try:
            x_letters_all, y_letters_all = extract_training_samples('letters')
            additional_mask = np.isin(y_letters_all, ADDITIONAL_LETTER_LABELS)
            x_letters = x_letters_all[additional_mask]
            x_letters = x_letters.astype('float32') / 255.0
            x_letters = x_letters.reshape(-1, 28, 28, 1)
            
            if len(x_letters) > n_letters_additional:
                indices = np.random.choice(len(x_letters), n_letters_additional, replace=False)
                x_letters = x_letters[indices]
            else:
                if len(x_letters) < n_letters_additional:
                    repeats = (n_letters_additional // len(x_letters)) + 1
                    x_letters = np.tile(x_letters, (repeats, 1, 1, 1))[:n_letters_additional]
            
            negative_images.append(x_letters)
            print(f"  Letters (Additional): {len(x_letters):,}")
        except Exception as e:
            print(f"  Warning: Could not load additional letters: {e}")
    
    # 9. Symbols (3%)
    if n_symbols > 0:
        symbols = ['+', '-', '=', '/', '*', '×', '÷', '(', ')', '[', ']', '{', '}', 
                  '|', '\\', ':', ';', '"', "'", '<', '>', '?', '!', 
                  '@', '#', '$', '%', '&', '^', '~', '`']
        symbol_images = []
        for i in range(n_symbols):
            symbol = np.random.choice(symbols)
            # Create 28x28 image with symbol
            img = np.zeros((28, 28), dtype=np.uint8)
            # Use OpenCV to draw symbol
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(symbol, font, font_scale, thickness)[0]
            text_x = (28 - text_size[0]) // 2
            text_y = (28 + text_size[1]) // 2
            cv2.putText(img, symbol, (text_x, text_y), font, font_scale, 255, thickness)
            # Normalize to [0, 1]
            img_normalized = img.astype(np.float32) / 255.0
            symbol_images.append(img_normalized)
        
        symbol_images = np.array(symbol_images).reshape(-1, 28, 28, 1)
        negative_images.append(symbol_images)
        print(f"  Symbols: {n_symbols:,}")
    
    # 10. Horizontal cut (2%)
    if x_digits is not None and n_horiz_cut > 0:
        if len(x_digits) >= n_horiz_cut:
            indices = np.random.choice(len(x_digits), n_horiz_cut, replace=False)
        else:
            # Use replacement if not enough digits
            indices = np.random.choice(len(x_digits), n_horiz_cut, replace=True)
        horiz_cut_images = x_digits[indices].copy()
        horiz_cut_images[:, 9:19, :] = 0
        negative_images.append(horiz_cut_images)
        print(f"  Horizontal Cut: {n_horiz_cut:,}")
    
    # 11. Vertical cut (2%)
    if x_digits is not None and n_vert_cut > 0:
        if len(x_digits) >= n_vert_cut:
            indices = np.random.choice(len(x_digits), n_vert_cut, replace=False)
        else:
            # Use replacement if not enough digits
            indices = np.random.choice(len(x_digits), n_vert_cut, replace=True)
        vert_cut_images = x_digits[indices].copy()
        vert_cut_images[:, :, 9:19] = 0
        negative_images.append(vert_cut_images)
        print(f"  Vertical Cut: {n_vert_cut:,}")
    
    # 12. Pixel removal (2%)
    if x_digits is not None and n_pixel_removal > 0:
        if len(x_digits) >= n_pixel_removal:
            indices = np.random.choice(len(x_digits), n_pixel_removal, replace=False)
        else:
            # Use replacement if not enough digits
            indices = np.random.choice(len(x_digits), n_pixel_removal, replace=True)
        pixel_removal_images = x_digits[indices].copy()
        for i in range(n_pixel_removal):
            img = pixel_removal_images[i].squeeze()
            nonzero_mask = img > 0.1
            nonzero_indices = np.where(nonzero_mask)
            n_nonzero = len(nonzero_indices[0])
            if n_nonzero > 0:
                n_to_remove = int(n_nonzero * 0.80)
                if n_nonzero >= n_to_remove:
                    remove_idx = np.random.choice(n_nonzero, n_to_remove, replace=False)
                else:
                    remove_idx = np.random.choice(n_nonzero, n_to_remove, replace=True)
                for idx in remove_idx:
                    pixel_removal_images[i, nonzero_indices[0][idx], nonzero_indices[1][idx], 0] = 0
        negative_images.append(pixel_removal_images)
        print(f"  Pixel Removal: {n_pixel_removal:,}")
    
    # 13. Custom non-digits (1%)
    x_custom = load_custom_non_digits()
    if len(x_custom) > 0:
        if len(x_custom) > n_custom:
            indices = np.random.choice(len(x_custom), n_custom, replace=False)
            x_custom = x_custom[indices]
        else:
            # Repeat/tile if we don't have enough custom images
            repeats = (n_custom // len(x_custom)) + 1
            x_custom = np.tile(x_custom, (repeats, 1, 1, 1))[:n_custom]
        negative_images.append(x_custom)
        print(f"  Custom Non-digits: {len(x_custom):,}")
    
    # ============================================================================
    # HARD NEGATIVES (40% = 16,920)
    # ============================================================================
    
    # Helper function to filter digits (exclude 1, 8 for partials; exclude 0 for rotations)
    def filter_digits_for_partials(x_digits, y_digits, count):
        """Filter to digits {0, 2, 3, 4, 5, 6, 7, 9} - exclude 1 and 8"""
        mask = np.isin(y_digits, [0, 2, 3, 4, 5, 6, 7, 9])
        x_filtered = x_digits[mask]
        y_filtered = y_digits[mask]
        if len(x_filtered) > count:
            indices = np.random.choice(len(x_filtered), count, replace=False)
            return x_filtered[indices], y_filtered[indices]
        elif len(x_filtered) > 0:
            # Repeat if we don't have enough
            repeats = (count // len(x_filtered)) + 1
            x_filtered = np.tile(x_filtered, (repeats, 1, 1, 1))[:count]
            y_filtered = np.tile(y_filtered, (repeats,))[:count]
            return x_filtered, y_filtered
        else:
            # Fallback: use all digits if filtered set is empty
            if len(x_digits) > count:
                indices = np.random.choice(len(x_digits), count, replace=False)
                return x_digits[indices], y_digits[indices]
            else:
                repeats = (count // len(x_digits)) + 1
                x_result = np.tile(x_digits, (repeats, 1, 1, 1))[:count]
                y_result = np.tile(y_digits, (repeats,))[:count]
                return x_result, y_result
    
    def filter_digits_for_multiple_cuts(x_digits, y_digits, count):
        """Filter to digits {0, 2, 3, 4, 5, 6, 7, 8, 9} - exclude 1"""
        mask = np.isin(y_digits, [0, 2, 3, 4, 5, 6, 7, 8, 9])
        x_filtered = x_digits[mask]
        y_filtered = y_digits[mask]
        if len(x_filtered) > count:
            indices = np.random.choice(len(x_filtered), count, replace=False)
            return x_filtered[indices], y_filtered[indices]
        elif len(x_filtered) > 0:
            # Repeat if we don't have enough
            repeats = (count // len(x_filtered)) + 1
            x_filtered = np.tile(x_filtered, (repeats, 1, 1, 1))[:count]
            y_filtered = np.tile(y_filtered, (repeats,))[:count]
            return x_filtered, y_filtered
        else:
            # Fallback: use all digits if filtered set is empty
            if len(x_digits) > count:
                indices = np.random.choice(len(x_digits), count, replace=False)
                return x_digits[indices], y_digits[indices]
            else:
                repeats = (count // len(x_digits)) + 1
                x_result = np.tile(x_digits, (repeats, 1, 1, 1))[:count]
                y_result = np.tile(y_digits, (repeats,))[:count]
                return x_result, y_result
    
    def filter_digits_for_rotation(x_digits, y_digits, count):
        """Filter for rotation: 40% from {6,9}, 40% from {2,3,5,7}, 20% from {1,4,8} - exclude 0"""
        n_69 = int(count * 0.40)  # ~340
        n_2357 = int(count * 0.40)  # ~340
        n_148 = count - n_69 - n_2357  # ~170
        
        result_x = []
        
        # 40% from {6, 9}
        mask_69 = np.isin(y_digits, [6, 9])
        x_69 = x_digits[mask_69]
        if len(x_69) > 0:
            if len(x_69) >= n_69:
                indices = np.random.choice(len(x_69), n_69, replace=False)
                result_x.append(x_69[indices])
            else:
                # Repeat if we don't have enough
                repeats = (n_69 // len(x_69)) + 1
                x_69 = np.tile(x_69, (repeats, 1, 1, 1))[:n_69]
                result_x.append(x_69)
        
        # 40% from {2, 3, 5, 7}
        mask_2357 = np.isin(y_digits, [2, 3, 5, 7])
        x_2357 = x_digits[mask_2357]
        if len(x_2357) > 0:
            if len(x_2357) >= n_2357:
                indices = np.random.choice(len(x_2357), n_2357, replace=False)
                result_x.append(x_2357[indices])
            else:
                # Repeat if we don't have enough
                repeats = (n_2357 // len(x_2357)) + 1
                x_2357 = np.tile(x_2357, (repeats, 1, 1, 1))[:n_2357]
                result_x.append(x_2357)
        
        # 20% from {1, 4, 8}
        mask_148 = np.isin(y_digits, [1, 4, 8])
        x_148 = x_digits[mask_148]
        if len(x_148) > 0:
            if len(x_148) >= n_148:
                indices = np.random.choice(len(x_148), n_148, replace=False)
                result_x.append(x_148[indices])
            else:
                # Repeat if we don't have enough
                repeats = (n_148 // len(x_148)) + 1
                x_148 = np.tile(x_148, (repeats, 1, 1, 1))[:n_148]
                result_x.append(x_148)
        
        if len(result_x) == 0:
            # Fallback: use all digits if no filtered digits available
            if len(x_digits) >= count:
                indices = np.random.choice(len(x_digits), count, replace=False)
                return x_digits[indices]
            else:
                repeats = (count // len(x_digits)) + 1
                x_result = np.tile(x_digits, (repeats, 1, 1, 1))[:count]
                return x_result
        
        # Concatenate all groups
        if len(result_x) > 0:
            result = np.concatenate(result_x, axis=0)
        else:
            result = np.array([]).reshape(0, 28, 28, 1)
        
        # Ensure we have exactly count items
        if len(result) < count:
            # Use all available digits to fill the gap
            needed = count - len(result)
            if len(x_digits) >= needed:
                # Get digits not already used (if possible)
                used_mask = np.zeros(len(x_digits), dtype=bool)
                # Try to avoid duplicates, but if needed, use replacement
                if needed <= len(x_digits):
                    additional_indices = np.random.choice(len(x_digits), needed, replace=False)
                else:
                    additional_indices = np.random.choice(len(x_digits), needed, replace=True)
                additional = x_digits[additional_indices]
            else:
                # Repeat available digits
                repeats = (needed // len(x_digits)) + 1
                additional = np.tile(x_digits, (repeats, 1, 1, 1))[:needed]
            result = np.concatenate([result, additional], axis=0)
        elif len(result) > count:
            # Trim to exact count
            result = result[:count]
        
        return result
    
    # 14. Partial Top Half (3%) - exclude 1, 8
    if x_digits is not None and y_digits is not None and n_partial_top > 0:
        x_partial, y_partial = filter_digits_for_partials(x_digits, y_digits, n_partial_top)
        partial_images = x_partial.copy()
        partial_images[:, 14:, :] = 0  # Remove bottom half
        negative_images.append(partial_images)
        print(f"  Partial Top Half: {len(partial_images):,}")
    
    # 15. Partial Bottom Half (3%) - exclude 1, 8
    if x_digits is not None and y_digits is not None and n_partial_bottom > 0:
        x_partial, y_partial = filter_digits_for_partials(x_digits, y_digits, n_partial_bottom)
        partial_images = x_partial.copy()
        partial_images[:, :14, :] = 0  # Remove top half
        negative_images.append(partial_images)
        print(f"  Partial Bottom Half: {len(partial_images):,}")
    
    # 16. Partial Left Half (2%) - exclude 1, 8
    if x_digits is not None and y_digits is not None and n_partial_left > 0:
        x_partial, y_partial = filter_digits_for_partials(x_digits, y_digits, n_partial_left)
        partial_images = x_partial.copy()
        partial_images[:, :, 14:] = 0  # Remove right half
        negative_images.append(partial_images)
        print(f"  Partial Left Half: {len(partial_images):,}")
    
    # 17. Partial Right Half (2%) - exclude 1, 8
    if x_digits is not None and y_digits is not None and n_partial_right > 0:
        x_partial, y_partial = filter_digits_for_partials(x_digits, y_digits, n_partial_right)
        partial_images = x_partial.copy()
        partial_images[:, :, :14] = 0  # Remove left half
        negative_images.append(partial_images)
        print(f"  Partial Right Half: {len(partial_images):,}")
    
    # 18. Multiple Cuts (2%) - exclude 1
    if x_digits is not None and y_digits is not None and n_multiple_cuts > 0:
        x_multi, y_multi = filter_digits_for_multiple_cuts(x_digits, y_digits, n_multiple_cuts)
        multi_images = x_multi.copy()
        multi_images[:, 9:19, :] = 0  # Horizontal cut
        multi_images[:, :, 9:19] = 0  # Vertical cut
        negative_images.append(multi_images)
        print(f"  Multiple Cuts: {len(multi_images):,}")
    
    # 19. Random Erasures (3%)
    if x_digits is not None and n_random_erasures > 0:
        if len(x_digits) >= n_random_erasures:
            indices = np.random.choice(len(x_digits), n_random_erasures, replace=False)
        else:
            indices = np.random.choice(len(x_digits), n_random_erasures, replace=True)
        erasure_images = x_digits[indices].copy()
        for i in range(n_random_erasures):
            # 2-4 random patches
            n_patches = np.random.randint(2, 5)
            for _ in range(n_patches):
                h_start = np.random.randint(0, 20)
                w_start = np.random.randint(0, 20)
                h_size = np.random.randint(3, 8)
                w_size = np.random.randint(3, 8)
                h_end = min(h_start + h_size, 28)
                w_end = min(w_start + w_size, 28)
                erasure_images[i, h_start:h_end, w_start:w_end, :] = 0
        negative_images.append(erasure_images)
        print(f"  Random Erasures: {n_random_erasures:,}")
    
    # 20. Blurred Digits (3%)
    if x_digits is not None and n_blurred > 0:
        if len(x_digits) >= n_blurred:
            indices = np.random.choice(len(x_digits), n_blurred, replace=False)
        else:
            indices = np.random.choice(len(x_digits), n_blurred, replace=True)
        blurred_images = []
        for idx in indices:
            img = (x_digits[idx].squeeze() * 255).astype(np.uint8)
            sigma = np.random.uniform(1.0, 3.0)
            blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=sigma)
            blurred_normalized = (blurred.astype(np.float32) / 255.0).reshape(28, 28, 1)
            blurred_images.append(blurred_normalized)
        blurred_images = np.array(blurred_images)
        negative_images.append(blurred_images)
        print(f"  Blurred Digits: {n_blurred:,}")
    
    # 21. Very Faint Digits (2%)
    if x_digits is not None and n_faint > 0:
        if len(x_digits) >= n_faint:
            indices = np.random.choice(len(x_digits), n_faint, replace=False)
        else:
            indices = np.random.choice(len(x_digits), n_faint, replace=True)
        faint_images = x_digits[indices].copy()
        for i in range(n_faint):
            contrast = np.random.uniform(0.2, 0.4)
            faint_images[i] = faint_images[i] * contrast
        negative_images.append(faint_images)
        print(f"  Very Faint Digits: {n_faint:,}")
    
    # 22. Rotated 90° Clockwise (2%) - exclude 0
    if x_digits is not None and y_digits is not None and n_rotated_90 > 0:
        x_rot = filter_digits_for_rotation(x_digits, y_digits, n_rotated_90)
        rotated_images = []
        for img in x_rot:
            img_uint8 = (img.squeeze() * 255).astype(np.uint8)
            rotated = cv2.rotate(img_uint8, cv2.ROTATE_90_CLOCKWISE)
            rotated_normalized = (rotated.astype(np.float32) / 255.0).reshape(28, 28, 1)
            rotated_images.append(rotated_normalized)
        rotated_images = np.array(rotated_images)
        negative_images.append(rotated_images)
        print(f"  Rotated 90°: {n_rotated_90:,}")
    
    # 23. Rotated 180° (2%) - exclude 0
    if x_digits is not None and y_digits is not None and n_rotated_180 > 0:
        x_rot = filter_digits_for_rotation(x_digits, y_digits, n_rotated_180)
        rotated_images = []
        for img in x_rot:
            img_uint8 = (img.squeeze() * 255).astype(np.uint8)
            rotated = cv2.rotate(img_uint8, cv2.ROTATE_180)
            rotated_normalized = (rotated.astype(np.float32) / 255.0).reshape(28, 28, 1)
            rotated_images.append(rotated_normalized)
        rotated_images = np.array(rotated_images)
        negative_images.append(rotated_images)
        print(f"  Rotated 180°: {n_rotated_180:,}")
    
    # 24. Distorted/Warped (2%)
    if x_digits is not None and n_distorted > 0:
        try:
            from scipy.ndimage import map_coordinates, gaussian_filter
            if len(x_digits) >= n_distorted:
                indices = np.random.choice(len(x_digits), n_distorted, replace=False)
            else:
                indices = np.random.choice(len(x_digits), n_distorted, replace=True)
            distorted_images = []
            for idx in indices:
                img = x_digits[idx].squeeze()
                # Elastic deformation
                alpha = np.random.uniform(50, 100)
                sigma = np.random.uniform(5, 10)
                dx = gaussian_filter((np.random.rand(28, 28) * 2 - 1) * alpha, sigma)
                dy = gaussian_filter((np.random.rand(28, 28) * 2 - 1) * alpha, sigma)
                x, y = np.meshgrid(np.arange(28), np.arange(28))
                indices_coords = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
                distorted = map_coordinates(img, indices_coords, order=1).reshape(28, 28, 1)
                distorted = np.clip(distorted, 0, 1)
                distorted_images.append(distorted)
            distorted_images = np.array(distorted_images)
            negative_images.append(distorted_images)
            print(f"  Distorted/Warped: {n_distorted:,}")
        except ImportError:
            # Fallback: use blurred digits instead if scipy not available
            print(f"  Warning: scipy not available, using blurred digits as fallback for distorted")
            if len(x_digits) >= n_distorted:
                indices = np.random.choice(len(x_digits), n_distorted, replace=False)
            else:
                indices = np.random.choice(len(x_digits), n_distorted, replace=True)
            distorted_images = []
            for idx in indices:
                img = (x_digits[idx].squeeze() * 255).astype(np.uint8)
                sigma = np.random.uniform(2.0, 4.0)
                blurred = cv2.GaussianBlur(img, (7, 7), sigmaX=sigma)
                blurred_normalized = (blurred.astype(np.float32) / 255.0).reshape(28, 28, 1)
                distorted_images.append(blurred_normalized)
            distorted_images = np.array(distorted_images)
            negative_images.append(distorted_images)
            print(f"  Distorted/Warped (fallback): {n_distorted:,}")
    
    # 25. Overlapping Shapes (2%)
    if x_digits is not None and n_overlapping > 0:
        overlapping_images = []
        for i in range(n_overlapping):
            if len(x_digits) >= 2:
                idx1, idx2 = np.random.choice(len(x_digits), 2, replace=False)
            else:
                # If only 1 digit available, use it for both
                idx1, idx2 = 0, 0
            digit1 = x_digits[idx1].squeeze()
            digit2 = x_digits[idx2].squeeze()
            offset_x = np.random.randint(-5, 6)
            offset_y = np.random.randint(-5, 6)
            digit1_shifted = np.roll(np.roll(digit1, offset_y, axis=0), offset_x, axis=1)
            overlapped = np.maximum(digit1_shifted, digit2)
            overlapping_images.append(overlapped.reshape(28, 28, 1))
        overlapping_images = np.array(overlapping_images)
        negative_images.append(overlapping_images)
        print(f"  Overlapping Shapes: {n_overlapping:,}")
    
    # 26. Stroke Breaks (2%)
    if x_digits is not None and n_stroke_breaks > 0:
        try:
            from scipy.ndimage import label
            if len(x_digits) >= n_stroke_breaks:
                indices = np.random.choice(len(x_digits), n_stroke_breaks, replace=False)
            else:
                indices = np.random.choice(len(x_digits), n_stroke_breaks, replace=True)
            stroke_break_images = []
            for idx in indices:
                img = x_digits[idx].squeeze()
                labeled, num_features = label(img > 0.5)
                if num_features > 1:
                    features_to_keep = np.random.choice(
                        num_features, 
                        size=np.random.randint(1, min(3, num_features)), 
                        replace=False
                    )
                    mask = np.isin(labeled, features_to_keep + 1)
                    img[mask == False] = 0
                stroke_break_images.append(img.reshape(28, 28, 1))
            stroke_break_images = np.array(stroke_break_images)
            negative_images.append(stroke_break_images)
            print(f"  Stroke Breaks: {n_stroke_breaks:,}")
        except ImportError:
            # Fallback: use partial cuts instead if scipy not available
            print(f"  Warning: scipy not available, using partial cuts as fallback for stroke breaks")
            if len(x_digits) >= n_stroke_breaks:
                indices = np.random.choice(len(x_digits), n_stroke_breaks, replace=False)
            else:
                indices = np.random.choice(len(x_digits), n_stroke_breaks, replace=True)
            stroke_break_images = x_digits[indices].copy()
            # Remove random patches to simulate stroke breaks
            for i in range(n_stroke_breaks):
                n_patches = np.random.randint(1, 3)
                for _ in range(n_patches):
                    h_start = np.random.randint(5, 20)
                    w_start = np.random.randint(5, 20)
                    h_size = np.random.randint(2, 5)
                    w_size = np.random.randint(2, 5)
                    h_end = min(h_start + h_size, 28)
                    w_end = min(w_start + w_size, 28)
                    stroke_break_images[i, h_start:h_end, w_start:w_end, :] = 0
            negative_images.append(stroke_break_images)
            print(f"  Stroke Breaks (fallback): {n_stroke_breaks:,}")
    
    # 27. Hard Negatives (Mining) (5%) - placeholder
    # This requires iterative training - collect misclassifications from previous model
    # For now, generate placeholder noise (will be replaced in iterative training)
    if n_hard_mining > 0:
        # Placeholder: use dense noise as fallback
        # In practice, this should be loaded from previous training failures
        hard_mining_images = np.zeros((n_hard_mining, 28, 28, 1), dtype=np.float32)
        for i in range(n_hard_mining):
            coverage = np.random.uniform(0.30, 0.60)
            mask = np.random.random((28, 28)) < coverage
            hard_mining_images[i, :, :, 0] = mask.astype(np.float32)
        negative_images.append(hard_mining_images)
        print(f"  Hard Negatives (Mining - placeholder): {n_hard_mining:,}")
    
    # ============================================================================
    # COMBINE AND RETURN
    # ============================================================================
    
    if len(negative_images) == 0:
        print("  Warning: No negative images created!")
        return np.array([]).reshape(0, 28, 28, 1), np.array([])
    
    # Combine all negative images
    x_negative = np.concatenate(negative_images, axis=0)
    
    # If we're short, fill with additional noise-based negatives
    if len(x_negative) < target_count:
        needed = target_count - len(x_negative)
        print(f"  Warning: Short by {needed:,} negatives. Filling with dense noise...")
        # Fill with dense noise
        additional_noise = np.zeros((needed, 28, 28, 1), dtype=np.float32)
        for i in range(needed):
            coverage = np.random.uniform(0.40, 0.70)
            mask = np.random.random((28, 28)) < coverage
            additional_noise[i, :, :, 0] = mask.astype(np.float32)
        x_negative = np.concatenate([x_negative, additional_noise], axis=0)
    elif len(x_negative) > target_count:
        # Trim to exact target if we somehow got more
        x_negative = x_negative[:target_count]
    
    # Labels are all class 10 (not a digit) - use integer labels for sparse_categorical_crossentropy
    y_negative = np.full((len(x_negative),), 10, dtype=np.int32)
    
    # Shuffle
    indices = np.random.permutation(len(x_negative))
    x_negative = x_negative[indices]
    y_negative = y_negative[indices]
    
    print(f"\nTotal negatives created: {len(x_negative):,}")
    print(f"Target was: {target_count:,}")
    if len(x_negative) != target_count:
        print(f"  ERROR: Count mismatch! Expected {target_count:,}, got {len(x_negative):,}")
    
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


def create_digit_classifier_model():
    """
    Create a CNN model for digit classification with 11 classes (0-9 digits + 10 "not a digit").
    
    Uses deep model architecture with 3 conv layers, optimized for EMNIST (240k+ samples).
    Always uses softmax activation with sparse_categorical_crossentropy loss.
    
    Returns:
        Compiled Keras model
    """
    # Model capacity for EMNIST (240k+ samples)
    number_convolution_channels = 32
    number_convolution_channelsF = 64
    neurons_in_dense_layer = 128
    
    # Always use softmax with 11 classes (0-9 digits + 10 "not a digit")
    output_activation = 'softmax'
    loss_function = 'sparse_categorical_crossentropy'
    output_layer = layers.Dense(11, activation=output_activation)  # 11 classes
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
        output_layer  # 11 classes: 0-9 digits + 10 "not a digit"
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
    - EMNIST Digits (sampled according to SAMPLE_RATIO per epoch) - includes MNIST
    - ARDIS (used 100% every epoch)
    - USPS (used 100% every epoch)
    - Font Digits (used 100% every epoch)
    - Custom "1" digits (used 100% every epoch) - serif style with no base
    
    Note: MNIST is NOT loaded separately since EMNIST Digits already includes it.
    
    Args:
        use_sigmoid: Whether to prepare labels for sigmoid (affects font digit loading)
    
    Returns:
        Tuple of (x_sampled_train, y_sampled_train, x_full_train, y_full_train, x_test, y_test)
        - x_sampled_train/y_sampled_train: EMNIST (will be sampled according to SAMPLE_RATIO per epoch)
        - x_full_train/y_full_train: ARDIS + USPS + Fonts + CustomOne (used 100% every epoch)
        - x_test/y_test: Combined test set from all datasets
        Arrays are normalized to [0, 1] and reshaped to (samples, 28, 28, 1)
    """
    # Datasets that will be sampled according to SAMPLE_RATIO per epoch
    sampled_datasets = []
    sampled_names = []
    
    # Datasets that will be used 100% every epoch
    full_datasets = []
    full_names = []
    
    # Test datasets (always combined)
    test_datasets = []
    
    # =========================================================================
    # SAMPLED DATASETS (SAMPLE_RATIO per epoch): EMNIST Digits (includes MNIST)
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
    print(f"SAMPLED datasets ({SAMPLE_RATIO*100:.0f}% per epoch): {' + '.join(sampled_names)}")
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
train_model=True, num_epochs=20):
    """
    Load a pre-trained digit classifier or create/train a new one.
    
    Always uses EMNIST (if available), data augmentation, and 4 layer CNN model.
    
    Args:
        classifier_model_path: Path to saved classifier model (.keras file)
        train_model: Whether to train a new model (True) or load existing (False)
        num_epochs: Number of training epochs (default: 20)
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
    
    # Create new model (always uses softmax with 11 classes)
    print("Creating new digit classifier model with 11 classes (0-9 digits + 10 'not a digit')...")
    model = create_digit_classifier_model()
    
    # Try to train on all digit datasets
    try:
        # Load and combine datasets
        # - sampled: EMNIST (will be sampled according to SAMPLE_RATIO per epoch)
        # - full: ARDIS + USPS + Fonts + CustomOne (used 100% every epoch)
        x_sampled_train, y_sampled_train, x_full_train, y_full_train, x_test, y_test = \
            load_and_combine_datasets(use_sigmoid=False)  # Always use integer labels for softmax
        
        # Store for digit tracking
        x_digits_test = x_test.copy()
        y_digits_test = y_test.copy()
        
        # Total digit counts for reference
        total_sampled_digits = len(x_sampled_train)  # EMNIST (will be sampled according to SAMPLE_RATIO)
        total_full_digits = len(x_full_train)         # ARDIS + USPS + Fonts (100% used)
        total_digits = total_sampled_digits + total_full_digits
        
        print(f"\nDigit data breakdown:")
        print(f"  SAMPLED ({SAMPLE_RATIO*100:.0f}% per epoch): {total_sampled_digits:,} (EMNIST)")
        print(f"  FULL (100% every epoch): {total_full_digits:,} (ARDIS + USPS + Fonts + CustomOne)")
        print(f"  Total digits: {total_digits:,}")
        
        # Create negative examples (non-digits) labeled as class 10
        print("\nCreating negative examples for softmax training (11 classes)...")
        print(f"  Target ratio: {NEGATIVE_RATIO*100:.0f}% of digit samples")
        x_negative_train, y_negative_train = create_negative_examples(total_digits, target_ratio=NEGATIVE_RATIO)
        x_negative_test, y_negative_test = create_negative_examples(len(x_digits_test), target_ratio=NEGATIVE_RATIO)
        
        print(f"  Total negative examples - Train: {len(x_negative_train):,}, Test: {len(x_negative_test):,}")
        print(f"\nTraining data:")
        print(f"  Digits: {total_digits:,} ({total_sampled_digits:,} sampled + {total_full_digits:,} full)")
        print(f"  Non-digits (class 10): {len(x_negative_train):,}")
        print(f"Test data: {len(x_digits_test):,} digits + {len(x_negative_test):,} non-digits")
        
        # Print per-class distribution for digits (0-9)
        print(f"\n=== Per-Class Distribution (Training) ===")
        y_all_digits = np.concatenate([y_sampled_train, y_full_train], axis=0)
        unique_digits, digit_counts = np.unique(y_all_digits, return_counts=True)
        for digit, count in zip(unique_digits, digit_counts):
            if digit < 10:  # Only digits 0-9
                print(f"  Digit {digit}: {count:,} samples ({count/total_digits*100:.1f}% of digits)")
        print(f"  Non-digits (class 10): {len(x_negative_train):,} samples")
        print(f"  Ratio: {len(x_negative_train)/total_digits:.2f} (non-digits/digits)")
        print(f"  Per-digit average: {total_digits/10:.0f} samples per digit class")
        print(f"  Non-digits vs per-digit: {len(x_negative_train)/(total_digits/10):.2f}x")
        print(f"==========================================\n")
        
        # Combine negatives with digits for training (negatives labeled as 10)
        # For sampled digits (EMNIST) - will be sampled per epoch
        x_sampled_train = np.concatenate([x_sampled_train, x_negative_train], axis=0)
        y_sampled_train = np.concatenate([y_sampled_train, y_negative_train], axis=0)
        
        # For test data
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
        # - SAMPLED datasets (EMNIST): SAMPLE_RATIO sampled each epoch
        # - FULL datasets (ARDIS+USPS+Fonts+CustomOne): 100% used every epoch
        sampled_per_epoch = int(total_sampled_digits * sample_ratio)
        full_per_epoch = total_full_digits  # 100% of ARDIS, USPS, Fonts
        digits_per_epoch = sampled_per_epoch + full_per_epoch
        
        print(f"Digit sampling per epoch:")
        print(f"  SAMPLED (EMNIST): {sample_ratio*100:.0f}% of {total_sampled_digits:,} = {sampled_per_epoch:,}")
        print(f"  FULL (ARDIS+USPS+Fonts+CustomOne): 100% of {total_full_digits:,} = {full_per_epoch:,}")
        print(f"  Total digits per epoch: {digits_per_epoch:,}")
        
        # Negatives are now part of sampled_train, so they're included in the sampling
        # Total samples per epoch includes negatives (which are in sampled_train)
        total_samples_per_epoch = sampled_per_epoch + full_per_epoch
        
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
        
        # Add softmax 11-class diagnostics callback
        softmax_callback = Softmax11DiagnosticsCallback(x_test, y_test)
        callbacks_list.append(softmax_callback)
        print("Softmax 11-class diagnostics will be printed after each epoch")
        
        # steps_per_epoch based on total sampled data
        steps_per_epoch = total_samples_per_epoch // train_datagen.batch_size
        
        # Use flow for softmax mode (digits + negatives combined, negatives labeled as 10)
        data_generator = train_datagen.flow(
            x_sampled_train, y_sampled_train,  # EMNIST (SAMPLE_RATIO sampled)
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
            
            # Note: Negatives are now part of sampled_train (labeled as class 10)
            # They're included in the total count but not separately tracked
            
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
    Classify a single digit image using the CNN model with 11 classes.
    
    Args:
        classifier_model: Trained Keras model (11 classes: 0-9 digits + 10 "not a digit")
        digit_image: 28x28 greyscale image (numpy array)
    
    Returns:
        Tuple of (predicted_digit, confidence)
        - predicted_digit: int (0-9) or -1 if class 10 ("not a digit") is predicted
        - confidence: float (0.0-1.0) or 0.0 if rejected
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
    
    # Get predicted class (0-10)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    
    # If class 10 ("not a digit") is predicted, return -1 with the actual confidence
    # Frontend will display the image with -1 to show what was rejected
    if predicted_class == 10:
        return -1, confidence  # Return actual class 10 probability
    
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
    
    args = parser.parse_args()
        
    # Train the model (always uses softmax with 11 classes)
    print("Starting digit classifier training with 11 classes (0-9 digits + 10 'not a digit')...")
    model = load_or_create_digit_classifier(
        classifier_model_path=args.model_path, 
        train_model=args.train_model,
        num_epochs=args.epoch_count
    )
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

