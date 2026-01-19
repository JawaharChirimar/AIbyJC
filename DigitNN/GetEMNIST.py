#!/usr/bin/env python3
"""
GetEMNIST.py

Functions to load EMNIST dataset directly from zip file.
This bypasses the broken emnist package which has NumPy compatibility issues.
"""

import os
import zipfile
import gzip
import struct
import numpy as np


def load_emnist_from_zip(zip_path='~/.cache/emnist/emnist.zip', split='digits'):
    """
    Load EMNIST dataset directly from zip file, bypassing the broken emnist package.
    
    Args:
        zip_path: Path to emnist.zip file (default: ~/.cache/emnist/emnist.zip)
        split: Which split to load ('digits', 'letters', 'balanced', 'byclass', 'bymerge', 'mnist')
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) as numpy arrays
        Returns (None, None, None, None) if loading fails
    """
    zip_path = os.path.expanduser(zip_path)
    
    if not os.path.exists(zip_path):
        print(f"  EMNIST zip file not found at: {zip_path}")
        return None, None, None, None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Load training images
            train_images_gz = f'gzip/emnist-{split}-train-images-idx3-ubyte.gz'
            train_labels_gz = f'gzip/emnist-{split}-train-labels-idx1-ubyte.gz'
            test_images_gz = f'gzip/emnist-{split}-test-images-idx3-ubyte.gz'
            test_labels_gz = f'gzip/emnist-{split}-test-labels-idx1-ubyte.gz'
            
            # Extract and decompress training images
            with zf.open(train_images_gz) as f_gz:
                with gzip.GzipFile(fileobj=f_gz) as f:
                    magic = struct.unpack('>I', f.read(4))[0]
                    num_images = struct.unpack('>I', f.read(4))[0]
                    rows = struct.unpack('>I', f.read(4))[0]
                    cols = struct.unpack('>I', f.read(4))[0]
                    x_train = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
            
            # Extract and decompress training labels
            with zf.open(train_labels_gz) as f_gz:
                with gzip.GzipFile(fileobj=f_gz) as f:
                    magic = struct.unpack('>I', f.read(4))[0]
                    num_labels = struct.unpack('>I', f.read(4))[0]
                    y_train = np.frombuffer(f.read(), dtype=np.uint8)
            
            # Extract and decompress test images
            with zf.open(test_images_gz) as f_gz:
                with gzip.GzipFile(fileobj=f_gz) as f:
                    magic = struct.unpack('>I', f.read(4))[0]
                    num_images = struct.unpack('>I', f.read(4))[0]
                    rows = struct.unpack('>I', f.read(4))[0]
                    cols = struct.unpack('>I', f.read(4))[0]
                    x_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
            
            # Extract and decompress test labels
            with zf.open(test_labels_gz) as f_gz:
                with gzip.GzipFile(fileobj=f_gz) as f:
                    magic = struct.unpack('>I', f.read(4))[0]
                    num_labels = struct.unpack('>I', f.read(4))[0]
                    y_test = np.frombuffer(f.read(), dtype=np.uint8)
            
            return x_train, y_train, x_test, y_test
            
    except Exception as e:
        print(f"  Error loading EMNIST from zip: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
