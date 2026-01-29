#!/usr/bin/env python3
"""
GetMNIST.py

Functions to load MNIST dataset with support for 64x64 upscaling.
Uses TensorFlow/Keras to load the original 28x28, then upscales with LANCZOS.
"""

import numpy as np
from pathlib import Path
from PIL import Image

# Data directory for saving 64x64 versions
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"

MNIST_DIR = DATA_DIR / "MNIST"


def upscale_images_to_size(images, target_size):
    """
    Upscale batch of images to target_size x target_size using LANCZOS.
    
    Args:
        images: numpy array (N, H, W) uint8
        target_size: Target size (28 or 64)
    
    Returns:
        numpy array (N, target_size, target_size) uint8
    """
    upscaled = []
    for img in images:
        pil_img = Image.fromarray(img)
        upscaled_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        upscaled.append(np.array(upscaled_img))
    return np.array(upscaled, dtype=np.uint8)


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


def create_one_hot_labels(labels, num_classes=10):
    """Create one-hot encoded labels for sigmoid output."""
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1.0
    return one_hot


def load_mnist_size(target_size=28, force_regenerate=False):
    """
    Load MNIST dataset at target_size x target_size.
    
    First checks for cached version. If not found, loads original 28x28,
    processes (upscales/downscales) if needed, and saves for future use.
    
    If data already matches target_size, no processing is performed.
    
    Saves train/test files with both softmax and sigmoid labels (images stored once).
    
    Args:
        target_size: Target size (28 or 64, default: 28)
        force_regenerate: If True, regenerate even if cached version exists
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
        - x arrays: uint8, shape (N, target_size, target_size)
        - y arrays: int, shape (N,) with values 0-9 (softmax format)
    """
    MNIST_DIR.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create one-hot labels for sigmoid
    y_train_sigmoid = create_one_hot_labels(y_train)
    y_test_sigmoid = create_one_hot_labels(y_test)
    
    # Save files (images once, both label formats)
    print(f"  Saving to {MNIST_DIR}...")
    np.savez(train_file, x=x_train_scaled, y_softmax=y_train, y_sigmoid=y_train_sigmoid)
    np.savez(test_file, x=x_test_scaled, y_softmax=y_test, y_sigmoid=y_test_sigmoid)
    
    print(f"  Saved:")
    print(f"    {train_file.name} - x: {x_train_scaled.shape}, y_softmax: {y_train.shape}, y_sigmoid: {y_train_sigmoid.shape}")
    print(f"    {test_file.name} - x: {x_test_scaled.shape}, y_softmax: {y_test.shape}, y_sigmoid: {y_test_sigmoid.shape}")
    
    return x_train_scaled, y_train, x_test_scaled, y_test

# Backward compatibility alias
def load_mnist_64x64(force_regenerate=False):
    """Backward compatibility wrapper for load_mnist_size(target_size=64)."""
    return load_mnist_size(target_size=64, force_regenerate=force_regenerate)


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
