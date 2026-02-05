#!/usr/bin/env python3
"""
PregenNonDigits.py

Pre-generates non-digit (negative) examples and saves them to disk.
This avoids loading EMNIST digits twice during training, saving ~10GB RAM.

Supports both 28x28 (default) and 64x64 image sizes.

Output files:
- data/non_digits_pregen/non_digits_train_augmented_{size}x{size}.npz (x, y where y=10 for all)
- data/non_digits_pregen/non_digits_test_augmented_{size}x{size}.npz (x, y where y=10 for all)
"""

import argparse
import sys
import numpy as np
import random
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Data directory
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"

NON_DIGITS_DIR = DATA_DIR / "non_digits_pregen"


def generate_and_save(force=False, image_size=28, train_count=47391, test_count=8809):
    """Generate non-digits and save to disk."""
    from DataManagement.NonDigitGenerator import create_negative_examples
    
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
        target_ratio=1.0,  # Generate exactly train_count samples
        image_size=image_size,
        dataset="Train"
    )
    
    # Save 50 sample images as PNG for inspection (randomly sampled for diversity)
    debug_dir = DATA_DIR / "non_digits_pregen" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    n_samples = min(50, len(x_neg_train))
    print(f"\nSaving {n_samples} randomly sampled non-digits to {debug_dir}/...")
    sample_indices = random.sample(range(len(x_neg_train)), n_samples)
    for idx, i in enumerate(sample_indices):
        # Convert from (H, W, 1) float32 [0,1] to (H, W) uint8 [0,255]
        img_array = (x_neg_train[i].squeeze() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(debug_dir / f"non_digit_sample_{idx:03d}.png")
    print(f"  Saved {n_samples} samples to {debug_dir}/")
    
    print(f"\nGenerating TEST non-digits ({test_count:,} samples)...")
    x_neg_test, y_neg_test = create_negative_examples(
        test_count, 
        target_ratio=1.0,  # Generate exactly test_count samples
        image_size=image_size,
        dataset="Test"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate non-digit examples")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Image size (28 or 64, default: 28)")
    parser.add_argument("--train-count", type=int, default=47391,
                        help="Number of training non-digits to generate (default: 47391)")
    parser.add_argument("--test-count", type=int, default=8809,
                        help="Number of test non-digits to generate (default: 8809)")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()
    
    generate_and_save(force=args.force, image_size=args.size, 
                     train_count=args.train_count, test_count=args.test_count)
