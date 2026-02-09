#!/usr/bin/env python3
"""
GetMNIST.py

Functions to load MNIST dataset with support for 64x64 upscaling.
Uses TensorFlow/Keras to load the original 28x28, then upscales with LANCZOS.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from DataManagement.DataCommon import (upscale_images_to_size, DATA_DIR)

MNIST_DIR = DATA_DIR / "MNIST"


def load_mnist_dataset():
    """
    Load MNIST dataset (28x28).
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
        - x arrays: uint8, shape (N, 28, 28)
        - y arrays: int, shape (N,) with values 0-9
    """
    try:
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        print(f"Loaded MNIST: {len(x_train)} training, {len(x_test)} test (28x28)")
        return x_train, y_train, x_test, y_test
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        return None, None, None, None


def load_mnist_size(target_size=28, force_regenerate=False):
    """
    Load MNIST dataset at target_size x target_size.
    
    First checks for cached version. If not found, loads original 28x28,
    processes (upscales/downscales) if needed, and saves for future use.
    
    If data already matches target_size, no processing is performed.
    
    Saves train/test files with both softmax labels (images stored once).
    
    Args:
        target_size: Target size (28 or 64, default: 28)
        force_regenerate: If True, regenerate even if cached version exists
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
        - x arrays: uint8, shape (N, target_size, target_size)
        - y arrays: int, shape (N,) with values 0-9 (softmax format)
    """
    # Cached file paths (train/test, images stored once with both label formats)
    train_file = MNIST_DIR / f"mnist_train_{target_size}x{target_size}.npz"
    test_file = MNIST_DIR / f"mnist_test_{target_size}x{target_size}.npz"
    
    # Try to load cached version
    if train_file.exists() and test_file.exists() and not force_regenerate:
        print(f"Loading cached MNIST {target_size}x{target_size}...")
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
    
    MNIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load original 28x28
    print(f"\nLoading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist_dataset()
    
    if x_train is None:
        return None, None, None, None
    
    # Check if data already matches target size
    current_size = x_train.shape[1] if len(x_train.shape) >= 2 else 28
    
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
    
    # Save files (images once, both label formats)
    print(f"  Saving to {MNIST_DIR}...")
    np.savez(train_file, x=x_train_scaled, y_softmax=y_train)
    np.savez(test_file, x=x_test_scaled, y_softmax=y_test)
    
    print(f"  Saved:")
    print(f"    {train_file.name} - x: {x_train_scaled.shape}, y_softmax: {y_train.shape}")
    print(f"    {test_file.name} - x: {x_test_scaled.shape}, y_softmax: {y_test.shape}")
    
    return x_train_scaled, y_train, x_test_scaled, y_test


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load MNIST dataset")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Target image size (28 or 64, default: 28)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if cached version exists")
    args = parser.parse_args()
    
    # Test the loaders
    print("Testing MNIST dataset loaders...")
    print("="*60)
    
    # Test 28x28 loader
    print("\n1. Testing 28x28 loader:")
    result = load_mnist_dataset()
    if result[0] is not None:
        x_train, y_train, x_test, y_test = result
        print(f"   Training: {x_train.shape}, labels: {np.unique(y_train)}")
        print(f"   Test: {x_test.shape}, labels: {np.unique(y_test)}")
    else:
        print("   MNIST 28x28 not available.")
    
    # Test size-specific loader
    print(f"\n2. Testing {args.size}x{args.size} loader:")
    result = load_mnist_size(target_size=args.size, force_regenerate=args.force)
    if result[0] is not None:
        x_train, y_train, x_test, y_test = result
        print(f"   Training: {x_train.shape}, labels: {np.unique(y_train)}")
        print(f"   Test: {x_test.shape}, labels: {np.unique(y_test)}")
    else:
        print("   MNIST 64x64 not available.")
    
    print("\n" + "="*60)
    print("Done!")
