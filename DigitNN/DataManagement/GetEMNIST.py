#!/usr/bin/env python3
"""
GetEMNIST.py

Functions to load EMNIST dataset using the emnist package.
The emnist package handles downloading, caching, and correct image orientation.

Supports both 28x28 (original) and 64x64 (upscaled with LANCZOS) versions.
"""

import numpy as np
from pathlib import Path
from DataManagement.DataCommon import (upscale_images_to_size, DATA_DIR)

EMNIST_DIR = DATA_DIR / "EMNIST"

# Check if emnist package is available
try:
    from emnist import extract_training_samples, extract_test_samples
    EMNIST_AVAILABLE = True
except ImportError:
    EMNIST_AVAILABLE = False
    print("Warning: 'emnist' package not available. Install with: pip install emnist")


def load_emnist_dataset(split='digits'):
    """
    Load EMNIST dataset using the emnist package.
    The emnist package handles downloading, caching, and correct image orientation.
    
    Args:
        split: Which split to load ('digits', 'letters', 'balanced', 'byclass', 'bymerge', 'mnist')
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) as numpy arrays
        - x arrays: uint8, shape (N, 28, 28)
        - y arrays: uint8, shape (N,)
        Returns (None, None, None, None) if loading fails
    """
    if not EMNIST_AVAILABLE:
        print("Error: emnist package not available. Install with: pip install emnist")
        return None, None, None, None
    
    try:
        print(f"Loading EMNIST {split} using emnist package...")
        
        # Load using emnist package (handles orientation correctly)
        x_train, y_train = extract_training_samples(split)
        x_test, y_test = extract_test_samples(split)
        
        # Convert to numpy arrays with correct dtype
        x_train = np.asarray(x_train, dtype=np.uint8)
        y_train = np.asarray(y_train, dtype=np.uint8)
        x_test = np.asarray(x_test, dtype=np.uint8)
        y_test = np.asarray(y_test, dtype=np.uint8)
        
        print(f"  Loaded: {len(x_train)} training, {len(x_test)} test (28x28)")
        return x_train, y_train, x_test, y_test
            
    except Exception as e:
        print(f"  Error loading EMNIST: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def load_emnist_size(split='digits', target_size=28, force_regenerate=False):
    """
    Load EMNIST dataset at target_size x target_size.
    
    First checks for cached version. If not found, loads original 28x28
    using emnist package, processes (upscales/downscales) if needed, and saves for future use.
    
    If data already matches target_size, no processing is performed.
    
    Saves train/test files with both softmax labels (images stored once).
    
    Args:
        split: Which split to load ('digits', 'letters', etc.)
        target_size: Target size (28 or 64, default: 28)
        force_regenerate: If True, regenerate even if cached version exists
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) as numpy arrays
        x arrays are (N, target_size, target_size) uint8
        y arrays are integer labels (softmax format)
    """
    # Cached file paths (train/test, images stored once with both label formats)
    train_file = EMNIST_DIR / f"emnist_{split}_train_{target_size}x{target_size}.npz"
    test_file = EMNIST_DIR / f"emnist_{split}_test_{target_size}x{target_size}.npz"
    
    # Try to load cached version
    if train_file.exists() and test_file.exists() and not force_regenerate:
        print(f"Loading cached EMNIST {target_size}x{target_size}...")
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

    EMNIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load original 28x28 using emnist package
    print(f"\nLoading EMNIST dataset...")
    x_train, y_train, x_test, y_test = load_emnist_dataset(split)
    
    if x_train is None:
        return None, None, None, None
    
    # Check if data already matches target size
    # EMNIST is always 28x28 from the package
    current_size = x_train.shape[1] if len(x_train.shape) >= 2 else 28
    
    if current_size == target_size:
        print(f"  Data is already {target_size}x{target_size}, no processing needed.")
        x_train_scaled = x_train
        x_test_scaled = x_test
    else:
        # Process (upscale or downscale) to target size
        print(f"  Processing training set from {current_size}x{current_size} to {target_size}x{target_size} with LANCZOS...")
        x_train_scaled = upscale_images_to_size(x_train, target_size)
        print(f"  Processing test set from {current_size}x{current_size} to {target_size}x{target_size} with LANCZOS...")
        x_test_scaled = upscale_images_to_size(x_test, target_size)
        
    # Save files (images once, both label formats)
    print(f"Saving to {EMNIST_DIR}...")
    np.savez(train_file, x=x_train_scaled, y_softmax=y_train)
    np.savez(test_file, x=x_test_scaled, y_softmax=y_test)
    
    print(f"  Saved:")
    print(f"    {train_file.name} - x: {x_train_scaled.shape}, y_softmax: {y_train.shape}")
    print(f"    {test_file.name} - x: {x_test_scaled.shape}, y_softmax: {y_test.shape}")
    
    return x_train_scaled, y_train, x_test_scaled, y_test


def load_emnist_letters_size(target_size=28, force=False):
    """
    Load or generate EMNIST letters (A-Z, a-z) at target_size x target_size resolution.
    
    Uses EMNIST ByClass to get 52 separate letter classes (uppercase and lowercase separate).
    Letters are labeled 10-61:
    - Uppercase A-Z: labels 10-35
    - Lowercase a-z: labels 36-61
    EMNIST letters are originally 28x28.
    
    Args:
        target_size: Target size (28 or 64, default: 28)
        force: If True, regenerate even if files exist
    
    Returns:
        Tuple of (x_train, y_train) or (None, None) if not available
        - x_train: (N, target_size, target_size) uint8
        - y_train: (N,) int32 (labels 10-61)
    """
    # Check if already generated
    letters_file = EMNIST_DIR / f"emnist_letters_train_{target_size}x{target_size}.npz"
    
    if letters_file.exists() and not force:
        print(f"Loading pre-generated EMNIST letters from {letters_file}")
        data = np.load(letters_file)
        return data['x'], data['y']
    
    # Generate
    EMNIST_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Extracting EMNIST letters using emnist package (byclass for 52 separate classes)...")
    x_train, y_train, _, _ = load_emnist_dataset('byclass')

    if x_train is None or y_train is None:
        return None, None
    
    # Filter to get only letters: labels 10-61 (uppercase 10-35, lowercase 36-61)
    letter_mask = (y_train >= 10) & (y_train <= 61)
    x_train = x_train[letter_mask]
    y_train = y_train[letter_mask]
    
    print(f"  Loaded {len(x_train):,} letter samples (52 classes: uppercase 10-35, lowercase 36-61)")
    print(f"  Unique labels: {np.unique(y_train)}")
    
    # Check if data already matches target size (EMNIST letters are 28x28)
    current_size = x_train.shape[1] if len(x_train.shape) >= 2 else 28
    
    if current_size == target_size:
        print(f"  Data is already {target_size}x{target_size}, no processing needed.")
        x_train_scaled = x_train
    else:
        # Process (upscale or downscale) to target size
        print(f"  Processing letters from {current_size}x{current_size} to {target_size}x{target_size} with LANCZOS...")
        x_train_scaled = upscale_images_to_size(x_train, target_size)
    
    # Save
    print(f"Saving to {letters_file}...")
    np.savez(letters_file, x=x_train_scaled, y=y_train)
    print(f"  Saved: x={x_train_scaled.shape}, y={y_train.shape}")
    
    return x_train_scaled, y_train


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load EMNIST dataset")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Target image size (28 or 64, default: 28)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if cached version exists")
    args = parser.parse_args()
    
    # Test the loaders
    print("Testing EMNIST dataset loaders...")
    print("="*60)
    
    # Test 28x28 loader
    print("\n1. Testing 28x28 loader (using emnist package):")
    result = load_emnist_dataset()
    if result[0] is not None:
        x_train, y_train, x_test, y_test = result
        print(f"   Training: {x_train.shape}, labels: {np.unique(y_train)}")
        print(f"   Test: {x_test.shape}, labels: {np.unique(y_test)}")
    else:
        print("   EMNIST 28x28 not available.")
    
    # Test size-specific loader
    print(f"\n2. Testing {args.size}x{args.size} loader:")
    result = load_emnist_size(target_size=args.size, force_regenerate=args.force)
    if result[0] is not None:
        x_train, y_train, x_test, y_test = result
        print(f"   Training: {x_train.shape}, labels: {np.unique(y_train)}")
        print(f"   Test: {x_test.shape}, labels: {np.unique(y_test)}")
    else:
        print(f"   EMNIST {args.size}x{args.size} not available.")
    
    # Test letters loader
    print(f"\n3. Testing letters {args.size}x{args.size} loader:")
    result = load_emnist_letters_size(target_size=args.size, force=args.force)
    if result[0] is not None:
        x_letters, y_letters = result
        print(f"   Letters: {x_letters.shape}, labels: {np.unique(y_letters)}")
    else:
        print(f"   EMNIST letters {args.size}x{args.size} not available.")
    
    print("\n" + "="*60)
    print("Done!")
