#!/usr/bin/env python3
"""
GetEMNIST.py

Functions to load EMNIST dataset using the emnist package.
The emnist package handles downloading, caching, and correct image orientation.

Supports both 28x28 (original) and 64x64 (upscaled with LANCZOS) versions.
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

EMNIST_DIR = DATA_DIR / "EMNIST"

# Check if emnist package is available
try:
    from emnist import extract_training_samples, extract_test_samples
    EMNIST_AVAILABLE = True
except ImportError:
    EMNIST_AVAILABLE = False
    print("Warning: 'emnist' package not available. Install with: pip install emnist")


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


# Alias for backwards compatibility
def load_emnist_from_zip(zip_path=None, split='digits'):
    """
    Backwards compatibility wrapper. Now uses emnist package instead of direct zip loading.
    The zip_path parameter is ignored.
    """
    return load_emnist_dataset(split=split)


def create_one_hot_labels(labels, num_classes=10):
    """Create one-hot encoded labels for sigmoid output."""
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1.0
    return one_hot


def load_emnist_size(split='digits', target_size=28, force_regenerate=False):
    """
    Load EMNIST dataset at target_size x target_size.
    
    First checks for cached version. If not found, loads original 28x28
    using emnist package, processes (upscales/downscales) if needed, and saves for future use.
    
    If data already matches target_size, no processing is performed.
    
    Saves train/test files with both softmax and sigmoid labels (images stored once).
    
    Args:
        split: Which split to load ('digits', 'letters', etc.)
        target_size: Target size (28 or 64, default: 28)
        force_regenerate: If True, regenerate even if cached version exists
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) as numpy arrays
        x arrays are (N, target_size, target_size) uint8
        y arrays are integer labels (softmax format)
    """
    EMNIST_DIR.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create one-hot labels for sigmoid
    y_train_sigmoid = create_one_hot_labels(y_train)
    y_test_sigmoid = create_one_hot_labels(y_test)
    
    # Save files (images once, both label formats)
    print(f"Saving to {EMNIST_DIR}...")
    np.savez(train_file, x=x_train_scaled, y_softmax=y_train, y_sigmoid=y_train_sigmoid)
    np.savez(test_file, x=x_test_scaled, y_softmax=y_test, y_sigmoid=y_test_sigmoid)
    
    print(f"  Saved:")
    print(f"    {train_file.name} - x: {x_train_scaled.shape}, y_softmax: {y_train.shape}, y_sigmoid: {y_train_sigmoid.shape}")
    print(f"    {test_file.name} - x: {x_test_scaled.shape}, y_softmax: {y_test.shape}, y_sigmoid: {y_test_sigmoid.shape}")
    
    return x_train_scaled, y_train, x_test_scaled, y_test

# Backward compatibility alias
def load_emnist_64x64(split='digits', force_regenerate=False):
    """Backward compatibility wrapper for load_emnist_size(target_size=64)."""
    return load_emnist_size(split=split, target_size=64, force_regenerate=force_regenerate)


def load_emnist_letters_size(target_size=28, force=False):
    """
    Load or generate EMNIST letters (A-Z) at target_size x target_size resolution.
    
    Letters are labeled 1-26 (A=1, B=2, ..., Z=26).
    EMNIST letters are originally 28x28.
    
    Args:
        target_size: Target size (28 or 64, default: 28)
        force: If True, regenerate even if files exist
    
    Returns:
        Tuple of (x_train, y_train) or (None, None) if not available
        - x_train: (N, target_size, target_size) uint8
        - y_train: (N,) int32 (labels 1-26)
    """
    if not EMNIST_AVAILABLE:
        print("EMNIST package not available.")
        return None, None
    
    # Check if already generated
    letters_file = EMNIST_DIR / f"emnist_letters_train_{target_size}x{target_size}.npz"
    
    if letters_file.exists() and not force:
        print(f"Loading pre-generated EMNIST letters from {letters_file}")
        data = np.load(letters_file)
        return data['x'], data['y']
    
    # Generate
    EMNIST_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Extracting EMNIST letters using emnist package...")
    x_train, y_train = extract_training_samples('letters')
    print(f"  Loaded {len(x_train):,} letter samples")
    print(f"  Labels: {np.unique(y_train)}")
    
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

# Backward compatibility alias
def load_emnist_letters_64x64(force=False):
    """Backward compatibility wrapper for load_emnist_letters_size(target_size=64)."""
    return load_emnist_letters_size(target_size=64, force=force)


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
