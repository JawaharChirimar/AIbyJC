#!/usr/bin/env python3
"""
GetUSPS.py

Functions to load and process the USPS dataset (US Postal Service handwritten digits).
USPS contains 9,298 digit images (16x16), resized to 28x28 to match MNIST/EMNIST.
"""

import bz2
import urllib.request
from pathlib import Path
import numpy as np
import cv2

# Data directory for USPS
DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"
USPS_DIR = DATA_DIR / "usps"


def load_usps_dataset():
    """
    Load USPS dataset (US Postal Service handwritten digits).
    Downloads automatically if not found locally.
    
    USPS contains 9,298 digit images (16x16), resized to 28x28 to match MNIST/EMNIST.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) or (None, None, None, None) if unavailable
        - x arrays: uint8, shape (N, 28, 28)
        - y arrays: int, shape (N,) with values 0-9
    """
    USPS_DIR.mkdir(parents=True, exist_ok=True)
    
    # USPS dataset from LIBSVM format (bz2 compressed)
    train_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2"
    test_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2"
    
    train_file = USPS_DIR / "usps.bz2"
    test_file = USPS_DIR / "usps.t.bz2"
    
    def download_file(url, filepath):
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
    
    def load_libsvm_bz2(filepath):
        """Load USPS LIBSVM format from bz2 file."""
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
                    
                    # Resize to 28x28 to match MNIST/EMNIST
                    img_28x28 = cv2.resize(img_16x16, (28, 28), interpolation=cv2.INTER_CUBIC)
                    images.append(img_28x28)
            
            return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)
        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")
            return None, None
    
    print("\nLoading USPS dataset...")
    
    # Download if needed
    if not download_file(train_url, train_file):
        return None, None, None, None
    if not download_file(test_url, test_file):
        return None, None, None, None
    
    # Load data
    x_train, y_train = load_libsvm_bz2(train_file)
    x_test, y_test = load_libsvm_bz2(test_file)
    
    if x_train is None or x_test is None:
        return None, None, None, None
    
    print(f"  Loaded USPS: {len(x_train)} training, {len(x_test)} test samples (resized 16x16 → 28x28)")
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    # Test the loader
    print("Testing USPS dataset loader...")
    result = load_usps_dataset()
    if result[0] is not None:
        x_train, y_train, x_test, y_test = result
        print(f"\nSummary:")
        print(f"  Training: {x_train.shape}, labels: {np.unique(y_train)}")
        print(f"  Test: {x_test.shape}, labels: {np.unique(y_test)}")
    else:
        print("\nUSPS dataset not available.")
