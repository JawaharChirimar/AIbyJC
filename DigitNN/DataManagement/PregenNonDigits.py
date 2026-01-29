#!/usr/bin/env python3
"""
PregenNonDigits.py

Pre-generates non-digit (negative) examples and saves them to disk.
This avoids loading EMNIST digits twice during training, saving ~10GB RAM.

Supports both 28x28 (default) and 64x64 image sizes.

Output files:
- data/non_digits_pregen/non_digits_train_{size}x{size}.npz (x, y where y=10 for all)
- data/non_digits_pregen/non_digits_test_{size}x{size}.npz (x, y where y=10 for all)
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# Data directory
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"

NON_DIGITS_DIR = DATA_DIR / "non_digits_pregen"


def generate_and_save(force=False, image_size=28):
    """Generate non-digits and save to disk."""
    from DataManagement.NonDigitGenerator import create_negative_examples, NEGATIVE_RATIO
    
    NON_DIGITS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_file = NON_DIGITS_DIR / f"non_digits_train_{image_size}x{image_size}.npz"
    test_file = NON_DIGITS_DIR / f"non_digits_test_{image_size}x{image_size}.npz"
    
    if train_file.exists() and test_file.exists() and not force:
        print("Non-digits already exist. Use --force to regenerate.")
        # Load and return info
        train_data = np.load(train_file)
        test_data = np.load(test_file)
        print(f"  Train: {train_data['x'].shape}")
        print(f"  Test: {test_data['x'].shape}")
        return
    
    # Count training digits (for target ratio)
    # Approximate based on typical dataset sizes
    # EMNIST: ~240K, ARDIS: ~6K, USPS: ~7K, Fonts: ~91K, Custom: ~4K = ~348K
    # We'll use actual EMNIST train count as base
    train_digit_count = 350000  # Approximate total training digits
    test_digit_count = 70000    # Approximate total test digits
    
    print(f"\nGenerating TRAINING non-digits (~{int(train_digit_count * NEGATIVE_RATIO):,} samples)...")
    x_neg_train, y_neg_train = create_negative_examples(
        train_digit_count, 
        target_ratio=NEGATIVE_RATIO,
        image_size=image_size
    )
    
    # Save 50 sample images as PNG for inspection
    debug_dir = DATA_DIR / "non_digits_pregen" / "samples"
    debug_dir.mkdir(parents=True, exist_ok=True)
    n_samples = min(50, len(x_neg_train))
    print(f"\nSaving {n_samples} sample non-digits to {debug_dir}/...")
    for i in range(n_samples):
        # Convert from (H, W, 1) float32 [0,1] to (H, W) uint8 [0,255]
        img_array = (x_neg_train[i].squeeze() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(debug_dir / f"non_digit_sample_{i:03d}.png")
    print(f"  Saved {n_samples} samples to {debug_dir}/")
    
    print(f"\nGenerating TEST non-digits (~{int(test_digit_count * NEGATIVE_RATIO):,} samples)...")
    x_neg_test, y_neg_test = create_negative_examples(
        test_digit_count, 
        target_ratio=NEGATIVE_RATIO,
        image_size=image_size
    )
    
    # Save
    print(f"\nSaving to {NON_DIGITS_DIR}...")
    
    # Convert to uint8 for storage (saves disk space)
    x_train_uint8 = (x_neg_train * 255).astype(np.uint8)
    x_test_uint8 = (x_neg_test * 255).astype(np.uint8)
    
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


def load_non_digits(image_size=28):
    """
    Load pre-generated non-digits from disk.
    
    Args:
        image_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) or (None, None, None, None)
        - x arrays: (N, image_size, image_size, 1) float32 [0,1]
        - y arrays: (N,) int32 (all values = 10)
    """
    train_file = NON_DIGITS_DIR / f"non_digits_train_{image_size}x{image_size}.npz"
    test_file = NON_DIGITS_DIR / f"non_digits_test_{image_size}x{image_size}.npz"
    
    if not train_file.exists() or not test_file.exists():
        print(f"Non-digits not found at {train_file}")
        print(f"Run PregenNonDigits.py --size {image_size} first!")
        return None, None, None, None
    
    train_data = np.load(train_file)
    test_data = np.load(test_file)
    
    x_train = train_data['x'].astype(np.float32) / 255.0
    y_train = train_data['y'].astype(np.int32)
    x_test = test_data['x'].astype(np.float32) / 255.0
    y_test = test_data['y'].astype(np.int32)
    
    # Ensure shape (N, image_size, image_size, 1)
    if len(x_train.shape) == 3:
        x_train = x_train.reshape(-1, image_size, image_size, 1)
        x_test = x_test.reshape(-1, image_size, image_size, 1)
    
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate non-digit examples")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Image size (28 or 64, default: 28)")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()
    
    generate_and_save(force=args.force, image_size=args.size)
