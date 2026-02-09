#!/usr/bin/env python3
"""
GetUSPS.py

Functions to load and process the USPS dataset (US Postal Service handwritten digits).
USPS contains 9,298 digit images (original size: 16x16).

Supports both 28x28 (default) and 64x64 (upscaled from 16x16 with LANCZOS interpolation).
"""

import bz2
import urllib.request
import numpy as np
from DataManagement.DataCommon import (upscale_images_to_size, DATA_DIR)

USPS_DIR = DATA_DIR / "usps"

# USPS dataset URLs (LIBSVM format, bz2 compressed)
_TRAIN_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2"
_TEST_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2"


def _download_file(url, filepath):
    """Download file if not exists."""
    if not filepath.exists():
        print(f"  Downloading {filepath.name}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  Downloaded: {filepath.name}")
        except Exception as e:
            print(f"  Error downloading {filepath.name}: {e}")
            return False
    return True


def _load_libsvm_bz2(filepath):
    """Load USPS LIBSVM format from bz2 file. Returns original 16x16."""
    try:
        images = []
        labels = []
        
        with bz2.open(filepath, 'rt') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                # Label is first (1-10 in USPS, convert to 0-9)
                label = int(float(parts[0])) - 1
                if label == -1:  # Label 10 means digit 0
                    label = 0
                labels.append(label)
                
                # Features are index:value pairs, values in [-1, 1]
                pixels = np.zeros(256, dtype=np.float32)
                for item in parts[1:]:
                    if ':' in item:
                        idx, val = item.split(':')
                        pixels[int(idx) - 1] = float(val)
                
                # Reshape to 16x16 and convert from [-1, 1] to [0, 255]
                img_16x16 = pixels.reshape(16, 16)
                img_16x16 = ((img_16x16 + 1) / 2 * 255).astype(np.uint8)
                images.append(img_16x16)
        
        return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)
    except Exception as e:
        print(f"  Error loading {filepath.name}: {e}")
        return None, None


def load_usps_dataset():
    """
    Load USPS dataset at original 16x16 size.
    Downloads automatically if not found locally.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) or (None, None, None, None) if unavailable
        - x arrays: uint8, shape (N, 16, 16)
        - y arrays: int, shape (N,) with values 0-9
    """
    USPS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_file = USPS_DIR / "usps.bz2"
    test_file = USPS_DIR / "usps.t.bz2"
    
    print("\nLoading USPS dataset...")
    
    # Download if needed
    if not _download_file(_TRAIN_URL, train_file):
        return None, None, None, None
    if not _download_file(_TEST_URL, test_file):
        return None, None, None, None
    
    # Load data
    x_train, y_train = _load_libsvm_bz2(train_file)
    x_test, y_test = _load_libsvm_bz2(test_file)
    
    if x_train is None or x_test is None:
        return None, None, None, None
    
    print(f"  Loaded USPS: {len(x_train)} training, {len(x_test)} test samples (16x16)")
    return x_train, y_train, x_test, y_test


def load_usps_size(target_size=28, force_regenerate=False):
    """
    Load USPS dataset at target_size x target_size.
    
    First checks for cached version. If not found, loads original 16x16,
    upscales with LANCZOS if needed, and saves for future use.
    
    Args:
        target_size: Target size (28 or 64, default: 28)
        force_regenerate: If True, regenerate even if cached version exists
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) or (None, None, None, None) if unavailable
        - x arrays: uint8, shape (N, target_size, target_size)
        - y arrays: int, shape (N,) with values 0-9 (softmax format)
    """
    # Cached file paths
    train_file = USPS_DIR / f"usps_train_{target_size}x{target_size}.npz"
    test_file = USPS_DIR / f"usps_test_{target_size}x{target_size}.npz"
    
    # Try to load cached version
    if train_file.exists() and test_file.exists() and not force_regenerate:
        print(f"Loading cached USPS {target_size}x{target_size}...")
        try:
            train_data = np.load(train_file)
            test_data = np.load(test_file)
            x_train = train_data['x']
            y_train = train_data['y_softmax']
            x_test = test_data['x']
            y_test = test_data['y_softmax']
            print(f"  Loaded: {len(x_train)} training, {len(x_test)} test ({target_size}x{target_size})")
            return x_train, y_train, x_test, y_test
        except Exception as e:
            print(f"  Error loading cache: {e}, regenerating...")
    
    USPS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load original 16x16
    x_train, y_train, x_test, y_test = load_usps_dataset()
    
    if x_train is None:
        return None, None, None, None
    
    # Check if data already matches target size
    current_size = x_train.shape[1] if len(x_train.shape) >= 2 else 16
    
    if current_size == target_size:
        print(f"  Data is already {target_size}x{target_size}, no processing needed.")
        x_train_scaled = x_train
        x_test_scaled = x_test
    else:
        # Process (upscale or downscale) to target size
        print(f"  Processing training set from {current_size}x{current_size} to {target_size}x{target_size}...")
        x_train_scaled = upscale_images_to_size(x_train, target_size)
        print(f"  Processing test set from {current_size}x{current_size} to {target_size}x{target_size}...")
        x_test_scaled = upscale_images_to_size(x_test, target_size)
    
    # Save files
    print(f"  Saving to {USPS_DIR}...")
    np.savez(train_file, x=x_train_scaled, y_softmax=y_train)
    np.savez(test_file, x=x_test_scaled, y_softmax=y_test)
    
    print(f"  Saved:")
    print(f"    {train_file.name} - x: {x_train_scaled.shape}, y_softmax: {y_train.shape}")
    print(f"    {test_file.name} - x: {x_test_scaled.shape}, y_softmax: {y_test.shape}")
    
    return x_train_scaled, y_train, x_test_scaled, y_test


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load USPS dataset")
    parser.add_argument("--size", type=int, default=28, choices=[16, 28, 64],
                        help="Target image size (16, 28, or 64, default: 28)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if cached version exists")
    args = parser.parse_args()
    
    # Test the loaders
    print("Testing USPS dataset loaders...")
    print("="*60)
    
    # Test original 16x16 loader
    print("\n1. Testing 16x16 loader (original):")
    result = load_usps_dataset()
    if result[0] is not None:
        x_train, y_train, x_test, y_test = result
        print(f"   Training: {x_train.shape}, labels: {np.unique(y_train)}")
        print(f"   Test: {x_test.shape}, labels: {np.unique(y_test)}")
    else:
        print("   USPS 16x16 not available.")
    
    # Test size-specific loader
    print(f"\n2. Testing {args.size}x{args.size} loader:")
    result = load_usps_size(target_size=args.size, force_regenerate=args.force)
    if result[0] is not None:
        x_train, y_train, x_test, y_test = result
        print(f"   Training: {x_train.shape}, labels: {np.unique(y_train)}")
        print(f"   Test: {x_test.shape}, labels: {np.unique(y_test)}")
    else:
        print(f"   USPS {args.size}x{args.size} not available.")
    
    print("\n" + "="*60)
    print("Done!")
