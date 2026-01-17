#!/usr/bin/env python3
"""
GetArdis.py

Functions to load and process the ARDIS dataset (Arkiv Digital Sweden handwritten digits).
ARDIS contains ~7,600 digit images from Swedish church records (1895-1970).
"""

from pathlib import Path
import numpy as np
import cv2

# Data directory for ARDIS
DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"
ARDIS_DIR = DATA_DIR / "ardis"


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


if __name__ == "__main__":
    # Test the loader
    print("Testing ARDIS dataset loader...")
    result = load_ardis_dataset()
    if result[0] is not None:
        x_train, y_train, x_test, y_test = result
        print(f"\nSummary:")
        print(f"  Training: {x_train.shape}, labels: {np.unique(y_train)}")
        print(f"  Test: {x_test.shape}, labels: {np.unique(y_test)}")
    else:
        print("\nARDIS dataset not available.")
