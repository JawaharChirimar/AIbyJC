#!/usr/bin/env python3
"""
PregenAugmentedBase.py

Common functions for pre-generating augmented dataset files.
Used by PregenAugmentedUSPS, PregenAugmentedArdis, PregenAugmentedEMNIST.
"""

import numpy as np
import math
import random
import sys
from collections import Counter
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import augmentation functions
from DataManagement.DataAugmentation import (
    apply_shear, apply_aspect_ratio,
    apply_blur, apply_thinning, apply_thickening,
    apply_random_pixel_erasure, apply_stroke_breaks,
    SHEAR_RANGE_POS, SHEAR_RANGE_NEG,
    ASPECT_WIDE_RANGE, ASPECT_NARROW_RANGE,
    BLUR_PROB, THIN_PROB, THICK_PROB, ERASURE_PROB, BREAKS_PROB,
    BLUR_RADIUS_RANGE
)

# Data directory
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"

# Augmentation ratio (10% of each class selected for augmentation)
AUGMENT_RATIO = 0.10
NUMBER_OF_AUGMENTATIONS = 4


def apply_post_processing(img_array, target_size, use_blur=True):
    """
    Apply post-processing with independent probability checks (matching PregenAugmentedData.py).
    Each effect is checked independently, so an image can get multiple effects.
    - 20% blur
    - 10% thin
    - 10% thick
    - 10% erasure (15% of white pixels)
    - 10% stroke breaks (6 × 4px)
    """
    result = img_array.copy()
    
    # Independent checks (matching PregenAugmentedData.py)
    if use_blur and random.random() < BLUR_PROB:
        radius = random.uniform(*BLUR_RADIUS_RANGE)
        result = apply_blur(result, radius)
    
    if random.random() < THIN_PROB:
        result = apply_thinning(result)
    else:
        if random.random() < THICK_PROB/(1.0-THIN_PROB):
            result = apply_thickening(result, target_size=target_size)
    
    if random.random() < ERASURE_PROB:
        result = apply_random_pixel_erasure(result)
    
    if random.random() < BREAKS_PROB:
        result = apply_stroke_breaks(result)
    
    return result


def augment_image_by_index(img_array, label, distortion_index, target_size):
    """
    Generate augmented versions of a single image (geometric transforms only, no post-processing).
    
    Args:
        img_array: numpy array of image
        label: integer label
        distortion_index: integer index of distortion to apply
        The distortion index is a number between 0 and NUMBER_OF_AUGMENTATIONS-1, where:
        0: Shear positive (+2° to +16°)
        1: Shear negative (-16° to -2°)
        2: Aspect wide (1.05 to 2.0)
        3: Aspect narrow (0.5 to 0.95)
        target_size: Target image size (28 or 64)
    
    Returns:
        A tuple (image, label) 
        image is the augmented image
        label is the label of the augmented image as passed in the function call
    """
    
    # 2. Shear positive (+2° to +16°)
    if distortion_index == 0:
        shear_pos = random.uniform(*SHEAR_RANGE_POS)
        sheared_pos = apply_shear(img_array, shear_pos, target_size=target_size)
        return (sheared_pos, label)
    
    # 3. Shear negative (-16° to -2°)
    if distortion_index == 1:
        shear_neg = random.uniform(*SHEAR_RANGE_NEG)
        sheared_neg = apply_shear(img_array, shear_neg, target_size=target_size)
        return (sheared_neg, label)
    
    # 4. Aspect wide (1.05 to 2.0)
    if distortion_index == 2:
        aspect_wide = random.uniform(*ASPECT_WIDE_RANGE)
        wide = apply_aspect_ratio(img_array, aspect_wide, target_size=target_size)
        return (wide, label)
    
    # 5. Aspect narrow (0.5 to 0.95)
    if distortion_index == 3:
        aspect_narrow = random.uniform(*ASPECT_NARROW_RANGE)
        narrow = apply_aspect_ratio(img_array, aspect_narrow, target_size=target_size)
        return (narrow, label)
    
    return None


def _load_dataset_npz(filepath, target_size):
    """Generic function to load dataset from npz file."""
    if not filepath.exists():
        return None, None
    
    data = np.load(filepath)
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.int32)

    if y is None or y.size == 0:
        raise ValueError(f"No labels found in {filepath}")

    if x is None or x.size == 0:
        raise ValueError(f"No images found in {filepath}")

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch: {x.shape[0]} images but {y.shape[0]} labels")
    
    if x.max() > 1.0:
        x = x / 255.0
    
    # Reshape if needed
    if len(x.shape) == 3:
        x = x.reshape(-1, x.shape[1], x.shape[2], 1)
    elif len(x.shape) == 4 and x.shape[-1] != 1:
        # If RGB, throw error (shouldn't happen for digit datasets)
        raise ValueError(f"Unexpected RGB images: shape {x.shape}, expected grayscale with channel dimension 1")
    
    # If image size doesn't match target, raise exception
    if x.shape[1] != target_size or x.shape[2] != target_size:
        raise ValueError(f"Image size mismatch: Images are {x.shape[1]}x{x.shape[2]}, expected {target_size}x{target_size}")
    
    return x, y


def load_dataset_data(dataset_name, dataset_dir, split='train', target_size=28, suffix=''):
    """
    Generic function to load un-augmented dataset.
    
    Args:
        dataset_name: Name prefix for files (e.g., "emnist_digits" or "ardis")
        dataset_dir: Path to dataset directory
        split: 'train' or 'test'
        target_size: Image size (28 or 64)
        suffix: Optional suffix before .npz
    """
    print(f"\n{'='*70}")
    print(f"Loading {dataset_name.upper()} {split.upper()} data ({target_size}x{target_size})")
    print(f"{'='*70}")
        
    # Load original data
    npz_file = dataset_dir / f"{dataset_name}_{split}_{target_size}x{target_size}{suffix}.npz"
    
    if not npz_file.exists():
        raise FileNotFoundError(f"{dataset_name} {split} data not found at {npz_file}")
    
    print(f"Loading {split} data from {npz_file}...")
    images, labels = _load_dataset_npz(npz_file, target_size)
    
    if images is None or labels is None:
        raise ValueError(f"Failed to load {dataset_name} {split} data")
    
    print(f"  Loaded {len(images):,} samples")
    print(f"  Labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}, len: {len(labels):,}")
    
    # Ensure labels is 1D numpy array
    if isinstance(labels, np.ndarray):
        if labels.ndim > 1:
            labels = labels.flatten()
    labels = np.array(labels).flatten()  # Ensure 1D
    
    # Count original distribution
    original_counts = Counter(labels)
    Original = {i: original_counts.get(i, 0) for i in range(10)}
    print(f"\nOriginal class distribution:")
    for i in range(10):
        print(f"  Class {i}: {Original[i]:,}")
    
    return images, labels
