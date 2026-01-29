#!/usr/bin/env python3
"""
GetArdis.py

Functions to load and process the ARDIS dataset (Arkiv Digital Sweden handwritten digits).
ARDIS contains ~7,600 digit images from Swedish church records (1895-1970).

Supports both 28x28 (original) and 64x64 (upscaled with LANCZOS) versions.
"""

from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Data directory for ARDIS
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"
    
ARDIS_DIR = DATA_DIR / "ardis"


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


def load_ardis_dataset():
    """
    Load ARDIS dataset (Arkiv Digital Sweden handwritten digits).
    
    ARDIS contains ~7,600 digit images from Swedish church records (1895-1970).
    Uses Dataset IV: 28x28 grayscale images (MNIST-style format).
    
    The dataset must be downloaded manually from:
    https://ardisdataset.github.io/ARDIS/
    
    Expected files in data/ardis/:
    - ARDIS_train_2828.csv (or images in folders 0-9)
    - ARDIS_test_2828.csv
    
    CSV format: Each row is [label, pixel1, pixel2, ..., pixel784]
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) or (None, None, None, None) if unavailable
        - x arrays: uint8, shape (N, 28, 28)
        - y arrays: int, shape (N,) with values 0-9
    """
    ARDIS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_file = ARDIS_DIR / "ARDIS_train_2828.csv"
    test_file = ARDIS_DIR / "ARDIS_test_2828.csv"
    
    # Also check for NPZ format (preprocessed)
    npz_file = ARDIS_DIR / "ardis.npz"
    
    def load_csv(filepath):
        """Load ARDIS CSV file. Format: label, pixel1, pixel2, ..., pixel784"""
        try:
            data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
            # First column is label, rest are pixels
            labels = data[:, 0].astype(np.int32)
            images = data[:, 1:].reshape(-1, 28, 28)
            # Convert to uint8 [0, 255]
            images = (images * 255).astype(np.uint8) if images.max() <= 1 else images.astype(np.uint8)
            return images, labels
        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")
            return None, None
    
    def load_from_folders():
        """Load ARDIS from folder structure (0-9 subfolders with images)."""
        images = []
        labels = []
        
        for digit in range(10):
            digit_folder = ARDIS_DIR / str(digit)
            if not digit_folder.exists():
                continue
            
            for img_path in digit_folder.glob("*"):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    try:
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Resize to 28x28 if needed
                            if img.shape != (28, 28):
                                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                            images.append(img)
                            labels.append(digit)
                    except Exception as e:
                        print(f"  Warning: Could not load {img_path}: {e}")
        
        if len(images) == 0:
            return None, None
        
        return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)
    
    print("\nLoading ARDIS dataset...")
    
    # Try NPZ first (fastest)
    if npz_file.exists():
        try:
            data = np.load(npz_file)
            x_train = data['x_train']
            y_train = data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']
            print(f"  Loaded ARDIS from NPZ: {len(x_train)} training, {len(x_test)} test samples")
            return x_train, y_train, x_test, y_test
        except Exception as e:
            print(f"  Warning: Could not load NPZ: {e}")
    
    # Try CSV files
    if train_file.exists() and test_file.exists():
        x_train, y_train = load_csv(train_file)
        x_test, y_test = load_csv(test_file)
        
        if x_train is not None and x_test is not None:
            print(f"  Loaded ARDIS from CSV: {len(x_train)} training, {len(x_test)} test samples")
            # Save as NPZ for faster loading next time
            np.savez(npz_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            print(f"  Saved to {npz_file} for faster loading")
            return x_train, y_train, x_test, y_test
    
    # Try folder structure
    x_all, y_all = load_from_folders()
    if x_all is not None and len(x_all) > 0:
        # Split 85% train, 15% test (similar to original ARDIS split)
        n_total = len(x_all)
        n_train = int(n_total * 0.85)
        indices = np.random.permutation(n_total)
        
        x_train = x_all[indices[:n_train]]
        y_train = y_all[indices[:n_train]]
        x_test = x_all[indices[n_train:]]
        y_test = y_all[indices[n_train:]]
        
        print(f"  Loaded ARDIS from folders: {len(x_train)} training, {len(x_test)} test samples")
        # Save as NPZ for faster loading next time
        np.savez(npz_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        print(f"  Saved to {npz_file} for faster loading")
        return x_train, y_train, x_test, y_test
    
    # Dataset not found - print instructions
    print("  ARDIS dataset not found!")
    print("  To use ARDIS, download Dataset IV from: https://ardisdataset.github.io/ARDIS/")
    print(f"  Then place files in: {ARDIS_DIR}")
    print("  Expected formats:")
    print("    - CSV: ARDIS_train_2828.csv, ARDIS_test_2828.csv")
    print("    - OR folders: 0/, 1/, 2/, ..., 9/ with digit images")
    return None, None, None, None


def create_one_hot_labels(labels, num_classes=10):
    """Create one-hot encoded labels for sigmoid output."""
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1.0
    return one_hot


def load_ardis_size(target_size=28, force_regenerate=False):
    """
    Load ARDIS dataset upscaled to target_size x target_size using LANCZOS.
    
    First checks for cached version. If not found, loads original 28x28,
    upscales with LANCZOS, and saves for future use.
    
    Saves train/test files with both softmax and sigmoid labels (images stored once).
    
    Args:
        target_size: Target size (28 or 64, default: 28)
        force_regenerate: If True, regenerate even if cached version exists
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) or (None, None, None, None) if unavailable
        - x arrays: uint8, shape (N, target_size, target_size)
        - y arrays: int, shape (N,) with values 0-9 (softmax format)
    """
    ARDIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cached file paths (train/test, images stored once with both label formats)
    train_file = ARDIS_DIR / f"ardis_train_{target_size}x{target_size}.npz"
    test_file = ARDIS_DIR / f"ardis_test_{target_size}x{target_size}.npz"
    
    # Try to load cached version
    if train_file.exists() and test_file.exists() and not force_regenerate:
        print(f"Loading cached ARDIS {target_size}x{target_size}...")
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
    print(f"\nLoading ARDIS dataset...")
    x_train, y_train, x_test, y_test = load_ardis_dataset()
    
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
    print(f"  Saving to {ARDIS_DIR}...")
    np.savez(train_file, x=x_train_scaled, y_softmax=y_train, y_sigmoid=y_train_sigmoid)
    np.savez(test_file, x=x_test_scaled, y_softmax=y_test, y_sigmoid=y_test_sigmoid)
    
    print(f"  Saved:")
    print(f"    {train_file.name} - x: {x_train_scaled.shape}, y_softmax: {y_train.shape}, y_sigmoid: {y_train_sigmoid.shape}")
    print(f"    {test_file.name} - x: {x_test_scaled.shape}, y_softmax: {y_test.shape}, y_sigmoid: {y_test_sigmoid.shape}")
    
    print(f"  Loaded ARDIS: {len(x_train_scaled)} training, {len(x_test_scaled)} test ({target_size}x{target_size})")
    return x_train_scaled, y_train, x_test_scaled, y_test

# Backward compatibility alias
def load_ardis_64x64(force_regenerate=False):
    """Backward compatibility wrapper for load_ardis_size(target_size=64)."""
    return load_ardis_size(target_size=64, force_regenerate=force_regenerate)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load ARDIS dataset")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Target image size (28 or 64, default: 28)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if cached version exists")
    args = parser.parse_args()
    
    # Test the loaders
    print("Testing ARDIS dataset loaders...")
    print("="*60)
    
    # Test 28x28 loader
    print("\n1. Testing 28x28 loader:")
    result = load_ardis_dataset()
    if result[0] is not None:
        x_train, y_train, x_test, y_test = result
        print(f"   Training: {x_train.shape}, labels: {np.unique(y_train)}")
        print(f"   Test: {x_test.shape}, labels: {np.unique(y_test)}")
    else:
        print("   ARDIS 28x28 not available.")
    
    # Test size-specific loader
    print(f"\n2. Testing {args.size}x{args.size} loader:")
    result = load_ardis_size(target_size=args.size, force_regenerate=args.force)
    if result[0] is not None:
        x_train, y_train, x_test, y_test = result
        print(f"   Training: {x_train.shape}, labels: {np.unique(y_train)}")
        print(f"   Test: {x_test.shape}, labels: {np.unique(y_test)}")
    else:
        print(f"   ARDIS {args.size}x{args.size} not available.")
    
    print("\n" + "="*60)
    print("Done!")
