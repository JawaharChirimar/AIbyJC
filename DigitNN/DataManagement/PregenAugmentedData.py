#!/usr/bin/env python3
"""
PregenAugmentedData.py

Pre-generates augmented training data to speed up training.
Instead of augmenting on-the-fly (slow), we pre-generate all augmented
images and save them to disk.

This reduces training time from ~2 hours/epoch to ~25 mins/epoch.

Supports both 28x28 (default) and 64x64 image sizes.

Output:
- data/augmented/train_augmented_{size}x{size}.npz (~3-5GB)
"""

import argparse
import numpy as np
from pathlib import Path
import time

# Data directory
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"

AUGMENTED_DIR = DATA_DIR / "augmented"


def load_all_training_data(image_size=28):
    """Load all training datasets (same as DigitClassifierSoftMax11.py)."""
    from DataManagement.PregenNonDigits import load_non_digits
    
    train_x_list = []
    train_y_list = []
    is_google_fonts_list = []
    dataset_names = []
    
    print(f"Loading all training datasets ({image_size}x{image_size})...")
    
    # 1. EMNIST
    emnist_path = DATA_DIR / "EMNIST" / f"emnist_digits_train_{image_size}x{image_size}.npz"
    if emnist_path.exists():
        data = np.load(emnist_path)
        x = data['x'].astype(np.float32)
        y = data['y_softmax'].astype(np.int32)
        if x.max() > 1.0:
            x = x / 255.0
        if len(x.shape) == 3:
            x = x.reshape(-1, image_size, image_size, 1)
        train_x_list.append(x)
        train_y_list.append(y)
        is_google_fonts_list.append(np.zeros(len(x), dtype=bool))
        dataset_names.append(f"EMNIST: {len(x):,}")
        print(f"  EMNIST: {len(x):,}")
    
    # 2. ARDIS
    ardis_path = DATA_DIR / "ardis" / f"ardis_train_{image_size}x{image_size}.npz"
    if ardis_path.exists():
        data = np.load(ardis_path)
        x = data['x'].astype(np.float32)
        y = data['y_softmax'].astype(np.int32)
        if x.max() > 1.0:
            x = x / 255.0
        if len(x.shape) == 3:
            x = x.reshape(-1, image_size, image_size, 1)
        train_x_list.append(x)
        train_y_list.append(y)
        is_google_fonts_list.append(np.zeros(len(x), dtype=bool))
        dataset_names.append(f"ARDIS: {len(x):,}")
        print(f"  ARDIS: {len(x):,}")
    
    # 3. USPS
    usps_path = DATA_DIR / "usps" / f"usps_train_{image_size}x{image_size}.npz"
    if usps_path.exists():
        data = np.load(usps_path)
        x = data['x'].astype(np.float32)
        y = data['y_softmax'].astype(np.int32)
        if x.max() > 1.0:
            x = x / 255.0
        if len(x.shape) == 3:
            x = x.reshape(-1, image_size, image_size, 1)
        train_x_list.append(x)
        train_y_list.append(y)
        is_google_fonts_list.append(np.zeros(len(x), dtype=bool))
        dataset_names.append(f"USPS: {len(x):,}")
        print(f"  USPS: {len(x):,}")
    
    # 4. Google Fonts (already augmented - mark as such)
    fonts_path = DATA_DIR / "font_digits" / f"font_digits_train_{image_size}x{image_size}_softmax.npz"
    if fonts_path.exists():
        data = np.load(fonts_path)
        x = data['x'].astype(np.float32)
        y = data['y'].astype(np.int32)
        if x.max() > 1.0:
            x = x / 255.0
        if len(x.shape) == 3:
            x = x.reshape(-1, image_size, image_size, 1)
        train_x_list.append(x)
        train_y_list.append(y)
        is_google_fonts_list.append(np.ones(len(x), dtype=bool))  # Mark as Google Fonts
        dataset_names.append(f"Google Fonts: {len(x):,}")
        print(f"  Google Fonts: {len(x):,} (skip augmentation)")
    
    # 5. Custom One
    custom_path = DATA_DIR / "custom_one" / f"custom_one_train_{image_size}x{image_size}_softmax.npz"
    if custom_path.exists():
        data = np.load(custom_path)
        x = data['x'].astype(np.float32)
        y = data['y'].astype(np.int32)
        if x.max() > 1.0:
            x = x / 255.0
        if len(x.shape) == 3:
            x = x.reshape(-1, image_size, image_size, 1)
        train_x_list.append(x)
        train_y_list.append(y)
        is_google_fonts_list.append(np.zeros(len(x), dtype=bool))
        dataset_names.append(f"Custom One: {len(x):,}")
        print(f"  Custom One: {len(x):,}")
    
    # 6. Non-digits (pre-generated)
    x_neg_train, y_neg_train, _, _ = load_non_digits(image_size=image_size)
    if x_neg_train is not None:
        train_x_list.append(x_neg_train)
        train_y_list.append(y_neg_train)
        is_google_fonts_list.append(np.zeros(len(x_neg_train), dtype=bool))
        dataset_names.append(f"Non-digits: {len(x_neg_train):,}")
        print(f"  Non-digits: {len(x_neg_train):,}")
    
    # Combine
    x_train = np.concatenate(train_x_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)
    is_google_fonts = np.concatenate(is_google_fonts_list, axis=0)
    
    print(f"\nTotal training samples: {len(x_train):,}")
    print(f"  Google Fonts (skip aug): {np.sum(is_google_fonts):,}")
    print(f"  To augment: {np.sum(~is_google_fonts):,}")
    
    return x_train, y_train, is_google_fonts


def augment_and_save(force=False, image_size=28):
    """Pre-generate augmented data and save to disk using DataAugmentation.py."""
    from DataManagement.DataAugmentation import ImageDataGeneratorWithAugmentation
    
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file = AUGMENTED_DIR / f"train_augmented_{image_size}x{image_size}.npz"
    
    if output_file.exists() and not force:
        print(f"Augmented data already exists at {output_file}")
        print("Use --force to regenerate.")
        data = np.load(output_file)
        print(f"  Shape: {data['x'].shape}")
        return
    
    # Load all training data
    x_train, y_train, is_google_fonts = load_all_training_data(image_size=image_size)
    
    print(f"\nUsing ImageDataGeneratorWithAugmentation from DataAugmentation.py")
    print(f"Total samples: {len(x_train):,}")
    
    # Create generator (uses same logic as on-the-fly augmentation)
    generator = ImageDataGeneratorWithAugmentation(
        x_train, y_train,
        is_google_fonts=is_google_fonts,
        augment_ratio=0.10,  # 10% selected for augmentation
        blur_prob=0.20,
        thin_prob=0.10,
        thick_prob=0.10,
        erasure_prob=0.10,
        breaks_prob=0.10
    )
    
    # Collect all augmented data from one epoch
    print(f"\nGenerating augmented data (one epoch)...")
    start_time = time.time()
    
    # Process all samples in one batch (already loaded in memory)
    batch_size = len(x_train)
    augmented_x = []
    augmented_y = []
    
    # Process exactly one epoch (one batch with all samples)
    for batch_x, batch_y in generator.flow(batch_size=batch_size, shuffle=True):
        augmented_x.append(batch_x)
        augmented_y.append(batch_y)
        break  # Only one batch needed since batch_size = len(x_train)
    
    # Concatenate all batches
    print("\nConcatenating batches...")
    augmented_x = np.concatenate(augmented_x, axis=0)
    augmented_y = np.concatenate(augmented_y, axis=0)
    
    # Shuffle
    print("Shuffling...")
    perm = np.random.permutation(len(augmented_x))
    augmented_x = augmented_x[perm]
    augmented_y = augmented_y[perm]
    
    # Save
    print(f"\nSaving to {output_file}...")
    
    # Convert to uint8 for smaller file size
    x_uint8 = (augmented_x * 255).astype(np.uint8)
    
    np.savez_compressed(output_file, x=x_uint8, y=augmented_y)
    
    total_time = time.time() - start_time
    file_size = output_file.stat().st_size / (1024 * 1024 * 1024)
    
    # Print stats
    generator.print_epoch_stats(epoch_num=1)
    
    print(f"\n{'='*60}")
    print(f"Augmentation complete!")
    print(f"{'='*60}")
    print(f"Total images: {len(augmented_x):,}")
    print(f"File size: {file_size:.2f} GB")
    print(f"Time: {total_time/60:.1f} minutes")
    print(f"Saved to: {output_file}")


def load_augmented_data(image_size=28):
    """
    Load pre-generated augmented training data.
    
    Args:
        image_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (x_train, y_train) or (None, None) if not found
        - x_train: (N, image_size, image_size, 1) float32 [0,1]
        - y_train: (N,) int32
    """
    output_file = AUGMENTED_DIR / f"train_augmented_{image_size}x{image_size}.npz"
    
    if not output_file.exists():
        print(f"Augmented data not found at {output_file}")
        print(f"Run PregenAugmentedData.py --size {image_size} first!")
        return None, None
    
    print(f"Loading pre-generated augmented data from {output_file}...")
    data = np.load(output_file)
    
    x_train = data['x'].astype(np.float32) / 255.0
    y_train = data['y'].astype(np.int32)
    
    if len(x_train.shape) == 3:
        x_train = x_train.reshape(-1, image_size, image_size, 1)
    
    print(f"  Loaded {len(x_train):,} samples")
    
    return x_train, y_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate augmented training data")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Image size (28 or 64, default: 28)")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()
    
    augment_and_save(force=args.force, image_size=args.size)
