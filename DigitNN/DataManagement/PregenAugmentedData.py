#!/usr/bin/env python3
"""
PregenAugmentedData.py

Loads pre-generated augmented training data from separate dataset files.
Each dataset has its own augmented file (EMNIST, ARDIS, USPS, Google Fonts, Non-digits).

This replaces the old approach of a single combined augmented file.
"""

import numpy as np
from pathlib import Path
from DataManagement.PregenAugmentedBase import DATA_DIR

def load_helper(data_path,dataset_name,image_size,x_list,y_list,dataset_names):
    if data_path.exists():
        try:
            data = np.load(data_path)
            x = data['x'].astype(np.float32)
            y = data['y'].astype(np.int32)

            if y is None or y.size == 0:
                raise ValueError(f"No labels found in {data_path}")

            if x is None or x.size == 0:
                raise ValueError(f"No images found in {data_path}")

            if x.shape[0] != y.shape[0]:
                raise ValueError(f"Mismatch: {x.shape[0]} images but {y.shape[0]} labels")
            
            # Normalize if needed (uint8 -> float32 [0,1])
            if x.max() > 1.0:
                x = x / 255.0
            
            # Reshape if needed
            if len(x.shape) == 3:
                x = x.reshape(-1, image_size, image_size, 1)
            
            x_list.append(x)
            y_list.append(y)
            dataset_names.append(f"{dataset_name}: {len(x):,}")
            print(f"  {dataset_name}: {len(x):,} samples")
        except Exception as e:
            print(f"  Warning: Could not load {dataset_name}: {e}")
    else:
        print(f"  Warning: {dataset_name} not found at {data_path}")


def load_augmented_data(image_size):
    """
    Load and combine all augmented training and test datasets from separate files.
    
    Loads from:
    Training:
    - EMNIST: emnist_digits_train_augmented_{size}x{size}.npz
    - ARDIS: ardis_train_augmented_{size}x{size}.npz
    - USPS: usps_train_augmented_{size}x{size}.npz
    - Google Fonts: font_digits_train_augmented_{size}x{size}.npz
    - Non-digits: non_digits_train_augmented_{size}x{size}.npz
    
    Test:
    - EMNIST: emnist_digits_test_augmented_{size}x{size}.npz
    - ARDIS: ardis_test_augmented_{size}x{size}.npz
    - USPS: usps_test_augmented_{size}x{size}.npz
    - Google Fonts: font_digits_test_augmented_{size}x{size}.npz
    - Non-digits: non_digits_test_augmented_{size}x{size}.npz
    
    Args:
        image_size: Image size (28 or 64)
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) or (None, None, None, None) if no data found
        - x_train: (N, image_size, image_size, 1) float32 [0,1]
        - y_train: (N,) int32
        - x_test: (M, image_size, image_size, 1) float32 [0,1]
        - y_test: (M,) int32
    """
    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []
    train_dataset_names = []
    test_dataset_names = []
    
    print(f"Loading augmented training data from separate files ({image_size}x{image_size})...")
    
    # =========================================================================
    # For training data
    # =========================================================================
    train_table = {
        "emnist": {"name": "EMNIST", "path1": "EMNIST", "path2": "emnist_digits_train_augmented"},
        "ardis": {"name": "ARDIS", "path1": "ardis", "path2": "ardis_train_augmented"},
        "usps": {"name": "USPS", "path1": "usps", "path2": "usps_train_augmented"},
        "google_fonts": {"name": "Google Fonts", "path1": "font_digits", "path2": "font_digits_train_augmented"},
        "non_digits": {"name": "Non-digits", "path1": "non_digits_pregen", "path2": "non_digits_train_augmented"}
    }

    for key in train_table:
        data_path = DATA_DIR / train_table[key]["path1"] / f"{train_table[key]['path2']}_{image_size}x{image_size}.npz"
        load_helper(data_path, train_table[key]["name"], image_size, train_x_list, train_y_list, train_dataset_names)
    

    # =========================================================================
    # For test data
    # =========================================================================
    test_table = {
        "emnist": {"name": "EMNIST", "path1": "EMNIST", "path2": "emnist_digits_test_augmented"},
        "ardis": {"name": "ARDIS", "path1": "ardis", "path2": "ardis_test_augmented"},
        "usps": {"name": "USPS", "path1": "usps", "path2": "usps_test_augmented"},
        "google_fonts": {"name": "Google Fonts", "path1": "font_digits", "path2": "font_digits_test_augmented"},
        "non_digits": {"name": "Non-digits", "path1": "non_digits_pregen", "path2": "non_digits_test_augmented"}
    }

    for key in test_table:
        data_path = DATA_DIR / test_table[key]["path1"] / f"{test_table[key]['path2']}_{image_size}x{image_size}.npz"
        load_helper(data_path, test_table[key]["name"], image_size, test_x_list, test_y_list, test_dataset_names)

    
    if len(train_x_list) == 0:
        print("\nERROR: No training datasets found!")
        print(f"Generate augmented training data first.")
        return None, None, None, None

    if len(test_x_list) == 0:
        print("\nERROR: No testing datasets found!")
        print(f"Generate augmented test data first.")
        return None, None, None, None
    
    # Combine all datasets
    x_train = np.concatenate(train_x_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)
    x_test = np.concatenate(test_x_list, axis=0)
    y_test = np.concatenate(test_y_list, axis=0)

    print(f"\nTotal training samples: {len(x_train):,}")
    print("Training datasets loaded:")
    for name in train_dataset_names:
        print(f"  - {name}")

    print(f"\nTotal testing samples: {len(x_test):,}")
    print("Testing datasets loaded:")
    for name in test_dataset_names:
        print(f"  - {name}")
    
    return x_train, y_train, x_test, y_test

