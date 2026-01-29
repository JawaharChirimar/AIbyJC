#!/usr/bin/env python3
"""
NonDigitGenerator.py

Creates negative examples (non-digit images) for training the digit classifier.
Generates various types of negatives:
- Easy: Black, White, Noise patterns
- Medium: Letters, Symbols, Cuts
- Hard: Partials, Blurred, Rotated, Distorted
"""

import cv2
import numpy as np
from pathlib import Path

# Import for pre-generated letters (size-aware)
from DataManagement.GetEMNIST import load_emnist_letters_size

# Import upscaling utility from DataAugmentation (if needed)
# Note: batch_upscale_to_64x64 is deprecated, use size-aware functions

# =============================================================================
# CONSTANTS
# =============================================================================
# image_size is now a parameter (default: 28)
NEGATIVE_RATIO = 0.20       # Ratio of negative examples to digit samples (20%)

# =============================================================================
# CACHE for EMNIST letters (avoid loading 4 times!)
# =============================================================================
_EMNIST_LETTERS_CACHE = None

def get_cached_emnist_letters(image_size=28):
    """Load EMNIST letters once and cache."""
    global _EMNIST_LETTERS_CACHE
    if _EMNIST_LETTERS_CACHE is None:
        from DataManagement.GetEMNIST import load_emnist_letters_size
        x_letters, y_letters = load_emnist_letters_size(target_size=image_size)
        if x_letters is not None:
            _EMNIST_LETTERS_CACHE = (x_letters, y_letters)
    return _EMNIST_LETTERS_CACHE

# Data directory
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"


def load_custom_non_digits(image_size=28):
    """
    Load custom non-digit images from CUSTOM_NON_DIGITS_DIR.
    Also loads individual files like not6.jpeg from the data directory.
    
    Args:
        image_size: Target image size (28 or 64, default: 28)
    
    Returns:
        numpy array of shape (N, image_size, image_size, 1), normalized [0,1], or empty array if none found
    """
    custom_images = []
    CUSTOM_NON_DIGITS_DIR = DATA_DIR / "non-digits"
    
    # Load individual not*.jpeg/png files from data directory
    for pattern in ["not*.jpeg", "not*.jpg", "not*.png"]:
        for img_path in DATA_DIR.glob(pattern):
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to image_size x image_size using LANCZOS
                    img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
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
                        img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
                        img_normalized = img_resized.astype(np.float32) / 255.0
                        custom_images.append(img_normalized)
                        print(f"  Loaded custom non-digit: {img_path.name}")
                except Exception as e:
                    print(f"  Warning: Could not load {img_path}: {e}")
    
    if custom_images:
        x_custom = np.array(custom_images)
        x_custom = x_custom.reshape(-1, image_size, image_size, 1)
        return x_custom
    
    return np.array([]).reshape(0, image_size, image_size, 1)


def create_negative_examples(total_digit_samples, target_ratio=NEGATIVE_RATIO, 
                             image_size=28):
    """
    Create negative examples for softmax training with 11 classes.
    These are images that are NOT digits, labeled as class 10.
    
    Generates negatives from:
    - Easy (25%): All Black, All White, Sparse/Dense Noise, Salt & Pepper, Gradient
    - Medium (50%): Letters (Safe + Additional), Symbols, Custom
    - Hard (25%): Hard Mining placeholder (noise)
    
    Args:
        total_digit_samples: Number of digit samples in training set
        target_ratio: Target ratio of negative examples (default NEGATIVE_RATIO = 20%)
        image_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (x_negative, y_negative) where:
        - x_negative: numpy array of shape (n, image_size, image_size, 1), normalized [0,1]
        - y_negative: numpy array of shape (n,), all labeled as 10 (not a digit)
    """
    target_count = int(total_digit_samples * target_ratio)
    
    # Allocate counts - redistributed to 100% (removed all digit-based negatives)
    # Easy (25%)
    n_black = int(target_count * 0.04)           # 4%
    n_white = int(target_count * 0.04)           # 4%
    n_sparse_noise = int(target_count * 0.06)    # 6%
    n_dense_noise = int(target_count * 0.05)      # 5%
    n_salt_pepper = int(target_count * 0.03)     # 3%
    n_gradient = int(target_count * 0.03)         # 3%
    
    # Medium (50%)
    n_letters_safe = int(target_count * 0.25)     # 25%
    n_letters_additional = int(target_count * 0.12)  # 12%
    n_symbols = int(target_count * 0.08)        # 8%
    n_custom = int(target_count * 0.05)          # 5%
    
    # Hard (25%)
    n_hard_mining = int(target_count * 0.25)     # 25% (placeholder - requires iterative training)
    
    # REMOVED: All digit-based negatives (cuts, partials, rotations, overlaps)
    # These created recognizable digits (0 with ghost, vertical line "1", etc.)
    
    negative_images = []
    
    print(f"\nCreating {target_count:,} negative examples ({target_ratio*100:.0f}% of {total_digit_samples:,} digits)...")
    
    # ============================================================================
    # EASY NEGATIVES (25%)
    # ============================================================================
    
    # 1. All black images (3%)
    if n_black > 0:
        black_images = np.zeros((n_black, image_size, image_size, 1), dtype=np.float32)
        negative_images.append(black_images)
        print(f"  All Black: {n_black:,}")
    
    # 2. All white images (3%)
    if n_white > 0:
        white_images = np.ones((n_white, image_size, image_size, 1), dtype=np.float32)
        negative_images.append(white_images)
        print(f"  All White: {n_white:,}")
    
    # 3. Sparse noise (5%)
    if n_sparse_noise > 0:
        sparse_noise_images = np.zeros((n_sparse_noise, image_size, image_size, 1), dtype=np.float32)
        for i in range(n_sparse_noise):
            coverage = np.random.uniform(0.05, 0.15)
            mask = np.random.random((image_size, image_size)) < coverage
            sparse_noise_images[i, :, :, 0] = mask.astype(np.float32)
        negative_images.append(sparse_noise_images)
        print(f"  Sparse Noise: {n_sparse_noise:,}")
    
    # 4. Dense noise (4%)
    if n_dense_noise > 0:
        dense_noise_images = np.zeros((n_dense_noise, image_size, image_size, 1), dtype=np.float32)
        for i in range(n_dense_noise):
            coverage = np.random.uniform(0.50, 0.80)
            mask = np.random.random((image_size, image_size)) < coverage
            dense_noise_images[i, :, :, 0] = mask.astype(np.float32)
        negative_images.append(dense_noise_images)
        print(f"  Dense Noise: {n_dense_noise:,}")
    
    # 5. Salt & Pepper noise (2%)
    if n_salt_pepper > 0:
        salt_pepper_images = np.zeros((n_salt_pepper, image_size, image_size, 1), dtype=np.float32)
        for i in range(n_salt_pepper):
            img = np.zeros((image_size, image_size))
            salt_pepper_mask = np.random.random((image_size, image_size)) < 0.1  # 10% pixels
            img[salt_pepper_mask] = np.random.choice([0.0, 1.0], size=np.sum(salt_pepper_mask))
            salt_pepper_images[i, :, :, 0] = img
        negative_images.append(salt_pepper_images)
        print(f"  Salt & Pepper: {n_salt_pepper:,}")
    
    # 6. Gradient noise (3%)
    if n_gradient > 0:
        gradient_images = np.zeros((n_gradient, image_size, image_size, 1), dtype=np.float32)
        for i in range(n_gradient):
            x = np.linspace(0, 1, image_size)
            y = np.linspace(0, 1, image_size)
            X, Y = np.meshgrid(x, y)
            gradient = (X + Y) / 2
            noise = np.random.random((image_size, image_size)) * 0.3
            img = np.clip(gradient + noise, 0, 1)
            gradient_images[i, :, :, 0] = img
        negative_images.append(gradient_images)
        print(f"  Gradient Noise: {n_gradient:,}")
    
    # ============================================================================
    # MEDIUM NEGATIVES (50%)
    # ============================================================================
    
    # REMOVED: EMNIST digits loading - no longer needed since we removed all digit-based negatives
    
    
    # REMOVED: All filter and helper functions for digit-based negatives
    # These are no longer needed since we removed all digit-based negative generation
    
    # 7. Letters - Safe (20%)
    # Removed: Y (label 25) - lowercase 'y' looks like 4 or 7
    # Removed: a (label 1), g (label 7), q (label 17) - too ambiguous
    SAFE_LETTER_LABELS = [8, 11, 13, 14, 18, 22, 23, 24]  # H, K, M, N, R, V, W, X
    if n_letters_safe > 0:
        try:
            # Load pre-generated letters directly from data/EMNIST (not cached)
            from DataManagement.GetEMNIST import load_emnist_letters_size
            x_letters_all, y_letters_all = load_emnist_letters_size(target_size=image_size)
            if x_letters_all is None:
                raise Exception("EMNIST letters not available")
            
            safe_mask = np.isin(y_letters_all, SAFE_LETTER_LABELS)
            x_letters = x_letters_all[safe_mask].copy()  # Copy to avoid modifying cache
            x_letters = x_letters.astype('float32') / 255.0
            x_letters = x_letters.reshape(-1, image_size, image_size, 1)
            
            if len(x_letters) > n_letters_safe:
                indices = np.random.choice(len(x_letters), n_letters_safe, replace=False)
                x_letters = x_letters[indices]
            else:
                if len(x_letters) < n_letters_safe:
                    repeats = (n_letters_safe // len(x_letters)) + 1
                    x_letters = np.tile(x_letters, (repeats, 1, 1, 1))[:n_letters_safe]
            
            # Already 64x64, no upscaling needed!
            negative_images.append(x_letters)
            print(f"  Letters (Safe): {len(x_letters):,}")
        except Exception as e:
            print(f"  Warning: Could not load safe letters: {e}")
    
    # 8. Letters - Additional (5%) - C, E, F, J, T, U (exclude O, I, S, Z, L, B, G, P, Q, D)
    # Removed: B (looks like 8), G (looks like 6/9), P (looks like 9), Q (looks like 9), D (looks like 0)
    ADDITIONAL_LETTER_LABELS = [3, 5, 6, 10, 20, 21]  # C, E, F, J, T, U
    if n_letters_additional > 0:
        try:
            # Load pre-generated letters directly from data/EMNIST (not cached)
            from DataManagement.GetEMNIST import load_emnist_letters_size
            x_letters_all, y_letters_all = load_emnist_letters_size(target_size=image_size)
            if x_letters_all is None:
                raise Exception("EMNIST letters not available")
            
            additional_mask = np.isin(y_letters_all, ADDITIONAL_LETTER_LABELS)
            x_letters = x_letters_all[additional_mask].copy()  # Copy to avoid modifying cache
            x_letters = x_letters.astype('float32') / 255.0
            x_letters = x_letters.reshape(-1, image_size, image_size, 1)
            
            if len(x_letters) > n_letters_additional:
                indices = np.random.choice(len(x_letters), n_letters_additional, replace=False)
                x_letters = x_letters[indices]
            else:
                if len(x_letters) < n_letters_additional:
                    repeats = (n_letters_additional // len(x_letters)) + 1
                    x_letters = np.tile(x_letters, (repeats, 1, 1, 1))[:n_letters_additional]
            
            # Already 64x64, no upscaling needed!
            negative_images.append(x_letters)
            print(f"  Letters (Additional): {len(x_letters):,}")
        except Exception as e:
            print(f"  Warning: Could not load additional letters: {e}")
    
    # 9. Symbols (3%)
    if n_symbols > 0:
        symbols = ['+', '=', '*', '÷', '(', ')', '[', ']', '{', '}', 
                   '"', "'", '<', '>', '?', 
                  '@', '#', '$', '%', '&', '^', '~']
        symbol_images = []
        for i in range(n_symbols):
            symbol = np.random.choice(symbols)
            # Create 64x64 image with symbol
            img = np.zeros((image_size, image_size), dtype=np.uint8)
            # Use OpenCV to draw symbol
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5  # Larger font for 64x64
            thickness = 3
            text_size = cv2.getTextSize(symbol, font, font_scale, thickness)[0]
            text_x = (image_size - text_size[0]) // 2
            text_y = (image_size + text_size[1]) // 2
            cv2.putText(img, symbol, (text_x, text_y), font, font_scale, 255, thickness)
            # Normalize to [0, 1]
            img_normalized = img.astype(np.float32) / 255.0
            symbol_images.append(img_normalized)
        
        symbol_images = np.array(symbol_images).reshape(-1, image_size, image_size, 1)
        negative_images.append(symbol_images)
        print(f"  Symbols: {n_symbols:,}")
        
    # 13. Custom non-digits (1%)
    x_custom = load_custom_non_digits(image_size=image_size)
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
    # HARD NEGATIVES (25%)
    # ============================================================================
        
    # 27. Hard Negatives (Mining) (5%) - placeholder
    # This requires iterative training - collect misclassifications from previous model
    # For now, generate placeholder noise (will be replaced in iterative training)
    if n_hard_mining > 0:
        # Placeholder: use dense noise as fallback
        # In practice, this should be loaded from previous training failures
        hard_mining_images = np.zeros((n_hard_mining, image_size, image_size, 1), dtype=np.float32)
        for i in range(n_hard_mining):
            coverage = np.random.uniform(0.30, 0.60)
            mask = np.random.random((image_size, image_size)) < coverage
            hard_mining_images[i, :, :, 0] = mask.astype(np.float32)
        negative_images.append(hard_mining_images)
        print(f"  Hard Negatives (Mining - placeholder): {n_hard_mining:,}")
    
    # ============================================================================
    # COMBINE AND RETURN
    # ============================================================================
    
    if len(negative_images) == 0:
        print("  Warning: No negative images created!")
        return np.array([]).reshape(0, image_size, image_size, 1), np.array([])
    
    # Combine all negative images
    x_negative = np.concatenate(negative_images, axis=0)
    
    # If we're short, fill with additional noise-based negatives
    if len(x_negative) < target_count:
        needed = target_count - len(x_negative)
        print(f"  Warning: Short by {needed:,} negatives. Filling with dense noise...")
        # Fill with dense noise
        additional_noise = np.zeros((needed, image_size, image_size, 1), dtype=np.float32)
        for i in range(needed):
            coverage = np.random.uniform(0.40, 0.70)
            mask = np.random.random((image_size, image_size)) < coverage
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


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NonDigitGenerator")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Image size (28 or 64, default: 28)")
    args = parser.parse_args()
    
    print("Testing NonDigitGenerator...")
    print(f"image_size: {args.size}")
    print(f"NEGATIVE_RATIO: {NEGATIVE_RATIO}")
    print(f"DATA_DIR: {DATA_DIR}")
    
    # Test with a small number
    x_neg, y_neg = create_negative_examples(10000, target_ratio=0.10, image_size=args.size)
    print(f"\nGenerated: {x_neg.shape}, labels: {y_neg.shape}")
    print(f"Label values: unique={np.unique(y_neg)}")
    print(f"Value range: min={x_neg.min():.3f}, max={x_neg.max():.3f}")
