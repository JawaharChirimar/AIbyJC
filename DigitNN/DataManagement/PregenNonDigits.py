#!/usr/bin/env python3
"""
PregenNonDigits.py

Pre-generate non-digit training and test data for image size 28 or 64.
"""

import argparse
import cv2
import numpy as np
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import for pre-generated letters (size-aware)
from DataManagement.GetEMNIST import load_emnist_letters_size

# Import augmentation functions
from DataManagement.PregenAugmentedBase import (
    augment_image_by_index,
    NUMBER_OF_AUGMENTATIONS,
    apply_post_processing
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Data directory
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"

NON_DIGITS_DIR = DATA_DIR / "non_digits_pregen"

CUSTOM_NON_DIGITS_DIR = DATA_DIR / "non-digits"
BLACK_WEIGHT = 0.02
WHITE_WEIGHT = 0.02
NOISE_WEIGHT = 0.05
GRADIENT_WEIGHT = 0.02
GEOMETRIC_WEIGHT = 0.05
SYMBOLS_WEIGHT = 0.16


def _empty_image_batch(image_size):
    """Return an empty image batch with the standard 4D shape."""
    return np.empty((0, image_size, image_size, 1), dtype=np.float32)


def load_custom_non_digits(image_size=28):
    """
    Load custom non-digit images from CUSTOM_NON_DIGITS_DIR.
    
    Args:
        image_size: Target image size (28 or 64, default: 28)
    
    Returns:
        numpy array of shape (N, image_size, image_size, 1), normalized [0,1], or empty array if none found
    """
    custom_images = []


    # Load from non-digits directory if it exists
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
    
    return _empty_image_batch(image_size)


def generate_noise(n_count, image_size):
    if n_count > 0:
        noise_images = np.zeros((n_count, image_size, image_size, 1), dtype=np.float32)
        for i in range(n_count):
            coverage = np.random.uniform(0.05, 0.80)
            mask = np.random.random((image_size, image_size)) < coverage
            noise_images[i, :, :, 0] = mask.astype(np.float32)

        print(f"  Noise: generated {len(noise_images):,}")
        print(f"  Noise: requested {n_count:,}")
        return noise_images
    else:
        return _empty_image_batch(image_size)


def generate_gradient(n_count, image_size):
    if n_count > 0: 
        gradient_images = np.zeros((n_count, image_size, image_size, 1), dtype=np.float32)
        for i in range(n_count):
            x = np.linspace(0, 1, image_size)
            y = np.linspace(0, 1, image_size)
            if np.random.random() > 0.5:
                x = 1 - x # Reverse the gradient
            if np.random.random() > 0.5:
                y = 1 - y # Reverse the gradient
            X, Y = np.meshgrid(x, y)
            gradient = (X + Y) / 2
            noise = np.random.random((image_size, image_size)) * 0.3
            img = np.clip(gradient + noise, 0, 1)
            gradient_images[i, :, :, 0] = img
        
        print(f"  Gradient Noise: generated {len(gradient_images):,}")
        print(f"  Gradient Noise: requested {n_count:,}")
        return gradient_images
    else:
        return _empty_image_batch(image_size)

def generate_geometric(n_count, image_size):
    if n_count > 0:
        geometric_images = np.zeros((n_count, image_size, image_size, 1), dtype=np.float32)
        for i in range(n_count):
            img = np.zeros((image_size, image_size), dtype=np.uint8)
            pattern = np.random.choice(['stripes', 'grid', 'crosshatch'])
            thickness = np.random.randint(1, 3)
            spacing = np.random.randint(4, 16)

            # Draw horizontal lines
            for pos in range(0, image_size, spacing):
                cv2.line(img, (0, pos), (image_size - 1, pos), 255, thickness)

            if pattern == 'grid':
                for pos in range(0, image_size, spacing):
                    cv2.line(img, (pos, 0), (pos, image_size - 1), 255, thickness)

            elif pattern == 'crosshatch':
                img2 = np.zeros((image_size, image_size), dtype=np.uint8)
                for pos in range(0, image_size, spacing):
                    cv2.line(img2, (0, pos), (image_size - 1, pos), 255, thickness)
                angle = np.random.uniform(30, 150)
                M = cv2.getRotationMatrix2D((image_size // 2, image_size // 2), angle, 1.0)
                img2 = cv2.warpAffine(img2, M, (image_size, image_size))
                img = cv2.max(img, img2)

            elif pattern == 'stripes':
                angle = np.random.uniform(0, 180)
                M = cv2.getRotationMatrix2D((image_size // 2, image_size // 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (image_size, image_size))

            geometric_images[i, :, :, 0] = img.astype(np.float32) / 255.0
        
        print(f"  Geometric: generated {len(geometric_images):,}")
        print(f"  Geometric: requested {n_count:,}")
        return geometric_images
    else:   
        return _empty_image_batch(image_size)

def generate_letters_safe(n_letters_safe, image_size):
    # Using EMNIST ByClass labels (10-61): uppercase 10-35, lowercase 36-61
    #[10, 12, 14, 15, 16, 17, 20, 22, 23, 25, 27, 29, 30, 31, 32, 33, 
    #37, 38, 39, 40, 41, 43, 46, 48, 49, 51, 53, 55, 56, 57, 58, 59]
    #[A, C, E, F, G, H, K, M, N, P, R, T, U, V, W, X, 
    #b, c, d, e, f, h, k, m, n, p, r, t, u, v, w, x]
    SAFE_LETTER_LABELS = [
        #A,  C,  E,  F,  G,  H,  K,  M,  N,
        10, 12, 14, 15, 16, 17, 20, 22, 23,
        #P,  R,  T,  U,  V,  W,  X, 
        25, 27, 29, 30, 31, 32, 33,
        #b,  c,  d,  e,  f,  h,  k,  m,  n,
        37, 38, 39, 40, 41, 43, 46, 48, 49,
        #p,  r,  t,  u,  v,  w,  x
        51, 53, 55, 56, 57, 58, 59]

    if  n_letters_safe > 0:
        try:
            # Load pre-generated letters directly from data/EMNIST (not cached)
            x_letters_all, y_letters_all = load_emnist_letters_size(target_size=image_size)
            if x_letters_all is None or y_letters_all is None:
                raise Exception("EMNIST letters not available")
            
            safe_mask = np.isin(y_letters_all, SAFE_LETTER_LABELS)
            x_letters = x_letters_all[safe_mask]
            x_letters = x_letters.astype('float32') / 255.0
            x_letters = x_letters.reshape(-1, image_size, image_size, 1)
            
            if len(x_letters) > n_letters_safe:
                indices = np.random.choice(len(x_letters), n_letters_safe, replace=False)
                x_letters = x_letters[indices]
            elif len(x_letters) < n_letters_safe:
                raise ValueError(f"Not enough safe letters: {len(x_letters)} < {n_letters_safe}")
            # Already 64x64, no upscaling needed!
            print(f"  Letters (Safe): generated {len(x_letters):,}")
            print(f"  Letters (Safe): requested {n_letters_safe:,}")
            return x_letters
        except Exception as e:
            raise ValueError(f"Could not load safe letters: {e}")
    else:   
        return _empty_image_batch(image_size)


def generate_symbols(n_symbols, image_size):
    if n_symbols > 0:
        symbols = ['+', '*', '÷', '(', ')', '[', ']', '{', '}', 
        '<', '>', '?', '£', '€', '¥', '¢', '§', '©', '®', '™', 
        '@', '#', '$', '%', '&', '^', '~']
        symbol_images = []
        
        # Load Unicode fonts from project fonts directory
        font_size = int(image_size * 0.90)  # Scale font with image size
        available_fonts = []
        
        project_fonts_dir = DATA_DIR / "fonts"
        if not project_fonts_dir.exists():
            raise FileNotFoundError(f"Fonts directory not found: {project_fonts_dir}. Please copy fonts to this directory.")
        
        for font_file in project_fonts_dir.glob("*.tt*"):
            try:
                font = ImageFont.truetype(str(font_file), font_size)
                available_fonts.append(font)
            except:
                print(f"  Warning: Could not load font: {font_file}")
                continue
        
        if len(available_fonts) == 0:
            raise ValueError(f"No valid fonts found in {project_fonts_dir}. Please ensure font files (.ttf or .ttc) are present.")
        
        for i in range(n_symbols):
            symbol = np.random.choice(symbols)
            # Create PIL image with symbol
            pil_img = Image.new('L', (image_size, image_size), 0)  # Black background
            draw = ImageDraw.Draw(pil_img)
            
            # Randomly select a font from available fonts for variety
            selected_font = np.random.choice(available_fonts)
            # Center the text using anchor='mm' (middle-middle)
            center_x = image_size // 2
            center_y = image_size // 2
            draw.text((center_x, center_y), symbol, fill=255, font=selected_font, anchor='mm')
            
            # Convert PIL image to numpy array and normalize
            img_array = np.array(pil_img, dtype=np.float32) / 255.0
            symbol_images.append(img_array)
        
        symbol_images = np.array(symbol_images).reshape(-1, image_size, image_size, 1)
        print(f"  Symbols: generated {len(symbol_images):,}")
        print(f"  Symbols: requested {n_symbols:,}")
        return symbol_images
    else:   
        return _empty_image_batch(image_size)


def augment_images(to_augment_images, target_size):
    augmented_images = []
    for img in to_augment_images:
        distortion_index = np.random.randint(0, NUMBER_OF_AUGMENTATIONS)
        aug_version = augment_image_by_index(img, 10, distortion_index, target_size=target_size)
        if aug_version is None:
            raise ValueError(f"Augmentation index {distortion_index} returned None for class 10")
        augmented_images.append(aug_version[0])

    return augmented_images


def create_negative_examples(target_count, 
                             image_size, dataset):
    """
    Create negative examples for training with 11 classes.
    These are images that are NOT digits, labeled as class 10.
    
    Args:
        target_count: +ve integer. Number of negative examples to generate.
        image_size: Image size (28 or 64)
        dataset: Dataset type (Train or Test)
    
    Returns:
        Tuple of (x_negative, y_negative) where:
        - x_negative: numpy array of shape (n, image_size, image_size, 1), normalized [0,1]
        - y_negative: numpy array of shape (n,), all labeled as 10 (not a digit)
    """

    n_black = int(target_count * BLACK_WEIGHT)           
    n_white = int(target_count * WHITE_WEIGHT)
    n_noise = int(target_count * NOISE_WEIGHT)    
    n_gradient = int(target_count * GRADIENT_WEIGHT)
    n_geometric = int(target_count * GEOMETRIC_WEIGHT)
    print(f"  n_black: {n_black:,}")
    print(f"  n_white: {n_white:,}")
    print(f"  n_noise: {n_noise:,}")
    print(f"  n_gradient: {n_gradient:,}")
    print(f"  n_geometric: {n_geometric:,}")

    n_custom = 0
    if CUSTOM_NON_DIGITS_DIR.exists() and dataset == "Train":
        image_extensions = ['.jpeg', '.jpg', '.png', '.bmp']
        n_custom = sum(1 for img_path in CUSTOM_NON_DIGITS_DIR.glob("*") 
                      if img_path.suffix.lower() in image_extensions)

    n_symbols = int(target_count * SYMBOLS_WEIGHT)

    print(f"  Total: {target_count:,}")

    n1 = n_black + n_white
    print(f"  n1 (sum of n_black + n_white): {n1:,}")

    mod3 = (target_count - n1) % 3
    print(f"  mod3 ((target_count - n1) % 3): {mod3:,}")
    
    n_white = n_white + mod3
    print(f"  n_white: {n_white:,}")

    n2 = (target_count - n1 - mod3)//3
    print(f"  n2 ((target_count - n1 - mod3)): {n2:,}")

    n_letters_safe = 2*n2 - n_symbols - n_custom - n_noise - n_gradient - n_geometric

    print(f"  Symbols: {n_symbols:,}")
    print(f"  Custom: {n_custom:,}")
    print(f"  Letters Safe: {n_letters_safe:,}")
        
    negative_images = []
    
    print(f"\nCreating {target_count:,} negative examples...")
        
    if n_black > 0:
        black_images = np.zeros((n_black, image_size, image_size, 1), dtype=np.float32)
        negative_images.append(black_images)
    
    if n_white > 0:
        white_images = np.ones((n_white, image_size, image_size, 1), dtype=np.float32)
        negative_images.append(white_images)
    
    to_augment_images = np.array([]).reshape(0, image_size, image_size, 1)
    if n_noise > 0:
        noise_images = generate_noise(n_noise, image_size)
        to_augment_images = np.concatenate([to_augment_images, noise_images], axis=0)

    if n_gradient > 0:
        gradient_images = generate_gradient(n_gradient, image_size)
        to_augment_images = np.concatenate([to_augment_images, gradient_images], axis=0)

    if n_geometric > 0:
        geometric_images = generate_geometric(n_geometric, image_size)
        to_augment_images = np.concatenate([to_augment_images, geometric_images], axis=0)
    
    if n_letters_safe > 0:
        x_letters = generate_letters_safe(n_letters_safe, image_size)
        to_augment_images = np.concatenate([to_augment_images, x_letters], axis=0)
    
    if n_symbols > 0:
        symbol_images = generate_symbols(n_symbols, image_size)
        to_augment_images = np.concatenate([to_augment_images, symbol_images], axis=0)
        
    # 13. Custom non-digits (1%)
    if n_custom > 0:
        x_custom = load_custom_non_digits(image_size=image_size)
        if len(x_custom) != n_custom:
            raise ValueError(f"{len(x_custom)} != {n_custom}")
        to_augment_images = np.concatenate([to_augment_images, x_custom], axis=0)
        print(f"  Custom Non-digits: {n_custom:,}")
    
    # Sample indices instead of arrays directly
    indices = np.random.choice(len(to_augment_images), n2, replace=False)
    picked_images = [to_augment_images[i] for i in indices]
    augmented_images = augment_images(picked_images, target_size=image_size)

    print(f"  Picked images: {len(picked_images):,}")
    print(f"  Augmented images: {len(augmented_images):,}")
    print(f"  Negative images: {len(negative_images):,}")
    print(f"  To augment images: {len(to_augment_images):,}")

    if len(negative_images) == 0:
        raise ValueError("No negative images created!")
        
    # Combine all negative images
    x_negative = np.concatenate([
        np.array(to_augment_images), 
        np.array(augmented_images), 
        np.concatenate(negative_images, axis=0)], axis=0)
    
    # Apply post-processing to all negative images (for consistency with digit images)
    print(f"\nApplying post-processing to {len(x_negative):,} negative images...")
    processed_negative = []
    for img in x_negative:
        processed = apply_post_processing(img, target_size=image_size)
        processed_negative.append(processed)
    x_negative = np.array(processed_negative)
    
    # If we're short, fill with additional noise-based negatives
    if len(x_negative) != target_count:
        raise ValueError(f"{len(x_negative)} != {target_count}")
    
    # Labels are all class 10 (not a digit) - use integer labels for sparse_categorical_crossentropy
    y_negative = np.full((len(x_negative),), 10, dtype=np.int32)
    
    # Shuffle
    indices = np.random.permutation(len(x_negative))
    x_negative = x_negative[indices]
    y_negative = y_negative[indices]
    
    print(f"\nTotal negatives created: {len(x_negative):,}")
    print(f"================================================")
    print(f"")
    
    return x_negative, y_negative

def generate_and_save(force, image_size, train_count, test_count):
    """Generate non-digits and save to disk."""
    
    NON_DIGITS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_file = NON_DIGITS_DIR / f"non_digits_train_augmented_{image_size}x{image_size}.npz"
    test_file = NON_DIGITS_DIR / f"non_digits_test_augmented_{image_size}x{image_size}.npz"
    
    if train_file.exists() and test_file.exists() and not force:
        print("Non-digits already exist. Use --force to regenerate.")
        # Load and return info
        train_data = np.load(train_file)
        test_data = np.load(test_file)
        print(f"  Train: {train_data['x'].shape}")
        print(f"  Test: {test_data['x'].shape}")
        return
    
    # Use exact counts (1:1 ratio with per-digit class counts)
    print(f"\nGenerating TRAINING non-digits ({train_count:,} samples)...")
    x_neg_train, y_neg_train = create_negative_examples(
        train_count, 
        image_size=image_size,
        dataset="Train"
    )
    
    # Save 50 sample images as PNG for inspection (randomly sampled for diversity)
    debug_dir = NON_DIGITS_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    n_samples = min(50, len(x_neg_train))
    print(f"\nSaving {n_samples} randomly sampled non-digits to {debug_dir}/...")
    sample_indices = np.random.choice(len(x_neg_train), n_samples, replace=False)
    for idx, i in enumerate(sample_indices):
        # Convert from (H, W, 1) float32 [0,1] to (H, W) uint8 [0,255]
        img_array = (x_neg_train[i].squeeze() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(debug_dir / f"non_digit_sample_{idx:03d}.png")
    print(f"  Saved {n_samples} samples to {debug_dir}/")
    
    print(f"\nGenerating TEST non-digits ({test_count:,} samples)...")
    x_neg_test, y_neg_test = create_negative_examples(
        test_count, 
        image_size=image_size,
        dataset="Test"
    )
    
    # Save
    print(f"\nSaving to {NON_DIGITS_DIR}...")
    
    # Convert to uint8 for storage (saves disk space) with safe range handling.
    x_train_uint8 = (np.clip(x_neg_train, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    x_test_uint8 = (np.clip(x_neg_test, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    
    np.savez_compressed(train_file, x=x_train_uint8, y=y_neg_train)
    np.savez_compressed(test_file, x=x_test_uint8, y=y_neg_test)
    
    # File sizes
    train_size = train_file.stat().st_size / (1024 * 1024)
    test_size = test_file.stat().st_size / (1024 * 1024)
    
    print(f"\nSaved:")
    print(f"  Train: {train_file} ({train_size:.1f} MB)")
    print(f"         Shape: {x_neg_train.shape}, Labels: all {np.unique(y_neg_train)}")
    print(f"  Test:  {test_file} ({test_size:.1f} MB)")
    print(f"         Shape: {x_neg_test.shape}, Labels: all {np.unique(y_neg_test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate non-digit training and test data")
    parser.add_argument("--size", type=int, choices=[28, 64], required=True,
                        help="Image size (28 or 64)")
    parser.add_argument("--train-count", type=int, required=True,
                        help="Number of training non-digits to generate")
    parser.add_argument("--test-count", type=int, required=True,
                        help="Number of test non-digits to generate")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration")
    args = parser.parse_args()

    if args.train_count < 1 or args.test_count < 1 or args.train_count < args.test_count:
        parser.error("train-count must be greater than test-count and both must be greater than 0")
    
    print(f"Generating non-digit training and test data for image size {args.size}...")
    print(f"  Training: {args.train_count:,} samples")
    print(f"  Testing: {args.test_count:,} samples")
    print(f"  Force: {args.force}")
    print(f"================================================")
    print(f"")
    generate_and_save(force=args.force, image_size=args.size, 
                     train_count=args.train_count, test_count=args.test_count)    