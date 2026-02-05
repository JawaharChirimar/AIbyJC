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
    apply_rotation, apply_shear, apply_aspect_ratio,
    apply_blur, apply_thinning, apply_thickening,
    apply_random_pixel_erasure, apply_stroke_breaks,
    ROTATION_RANGE_POS, ROTATION_RANGE_NEG,
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

def apply_post_processing(img_array):
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
    if random.random() < BLUR_PROB:
        radius = random.uniform(*BLUR_RADIUS_RANGE)
        result = apply_blur(result, radius)
    
    if random.random() < THIN_PROB:
        result = apply_thinning(result)
    else:
        if random.random() < THICK_PROB*10.0/9.0:
            result = apply_thickening(result)
    
    if random.random() < ERASURE_PROB:
        result = apply_random_pixel_erasure(result)
    
    if random.random() < BREAKS_PROB:
        result = apply_stroke_breaks(result)
    
    return result


def augment_image(img_array, label):
    """
    Generate augmented versions of a single image (geometric transforms only, no post-processing).
    
    Returns:
        List of (image, label) tuples: original + 5 augmented
        1. Original (no post-processing)
        2. Rotation (±3° to ±30°)
        3. Shear positive (+2° to +16°)
        4. Shear negative (-16° to -2°)
        5. Aspect wide (1.05 to 2.0)
        6. Aspect narrow (0.5 to 0.95)
    """
    results = []
    
    # 1. Original (no post-processing)
    results.append((img_array.copy(), label))
    
    # 2. Rotation (random + or -)
    angle = random.uniform(*ROTATION_RANGE_POS) if random.random() > 0.5 else random.uniform(*ROTATION_RANGE_NEG)
    rotated = apply_rotation(img_array, angle)
    results.append((rotated, label))
    
    # 3. Shear positive (+2° to +16°)
    shear_pos = random.uniform(*SHEAR_RANGE_POS)
    sheared_pos = apply_shear(img_array, shear_pos)
    results.append((sheared_pos, label))
    
    # 4. Shear negative (-16° to -2°)
    shear_neg = random.uniform(*SHEAR_RANGE_NEG)
    sheared_neg = apply_shear(img_array, shear_neg)
    results.append((sheared_neg, label))
    
    # 5. Aspect wide (1.05 to 2.0)
    aspect_wide = random.uniform(*ASPECT_WIDE_RANGE)
    wide = apply_aspect_ratio(img_array, aspect_wide)
    results.append((wide, label))
    
    # 6. Aspect narrow (0.5 to 0.95)
    aspect_narrow = random.uniform(*ASPECT_NARROW_RANGE)
    narrow = apply_aspect_ratio(img_array, aspect_narrow)
    results.append((narrow, label))
    
    return results


def calculate_balancing_parameters(Original):
    """
    Calculate augmentation percentages and counts using balancing algorithm.
    
    Args:
        Original: dict of {class_idx: count} for original class distribution
    
    Returns:
        Percent: dict of {class_idx: percent_to_augment}
        Augmented: dict of {class_idx: num_augmented_images_to_create}
        m: target count per class after balancing
    """
    import math
    
    Percent = {}
    PreAugmented = {}
    PreFinal = {}

    maxIndex = max(Original, key=Original.get)    
    # Step 1: Calculate Percent[maxIndex] and PreFinal[maxIndex]
    Percent[maxIndex] = 0.01
    PreAugmented[maxIndex] = int(math.floor(Original[maxIndex] * Percent[maxIndex])) * 5
    PreFinal[maxIndex] = Original[maxIndex] + PreAugmented[maxIndex]
    m = PreFinal[maxIndex]  # Initialize minimum
    
    # Step 2: Calculate Percent[k] for k = 1 to 9, and track minimum
    for k in range(0, 10):
        if k == maxIndex:
            continue
        Percent[k] = (PreFinal[maxIndex] - Original[k]) / (5 * Original[k])
        PreAugmented[k] = int(math.floor(Original[k] * Percent[k])) * 5
        PreFinal[k] = Original[k] + PreAugmented[k]
        m = min(m, PreFinal[k])  # Update minimum in the loop
    
    # Step 4: Calculate final augmented counts
    Augmented = {}
    for i in range(10):
        residue = PreFinal[i] - m        # Residue is the difference between the final count and the target count
        Augmented[i] = PreAugmented[i] - residue

    return Percent, Augmented, m


def load_dataset_npz(filepath, target_size):
    """Generic function to load dataset from npz file."""
    if not filepath.exists():
        return None, None
    
    data = np.load(filepath)
    x = data['x'].astype(np.float32)
    y = data['y_softmax'].astype(np.int32) if 'y_softmax' in data else data['y'].astype(np.int32)
    
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


def process_dataset_data(dataset_name, dataset_dir, split='train', target_size=28, force=False, 
                        augment_ratio=0.10, output_suffix=""):
    """
    Generic function to process dataset with augmentation.
    Automatically detects if classes are balanced and uses appropriate method:
    - If balanced: uses simple augment_ratio (same % for all classes)
    - If not balanced: uses balancing algorithm to balance classes
    
    Args:
        dataset_name: Name prefix for files (e.g., "emnist_digits" or "ardis")
        dataset_dir: Path to dataset directory
        split: 'train' or 'test'
        target_size: Image size (28 or 64)
        force: Force regeneration if file exists
        augment_ratio: Fraction of each class to augment (default 0.10, used if classes are balanced)
        output_suffix: Optional suffix for output filename (e.g., "_balanced")
    """
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name.upper()} {split.upper()} data ({target_size}x{target_size})")
    print(f"{'='*70}")
    
    # Output file
    suffix = output_suffix if output_suffix else ""
    output_file = dataset_dir / f"{dataset_name}_{split}_augmented{suffix}_{target_size}x{target_size}.npz"
    
    if output_file.exists() and not force:
        print(f"Augmented data already exists at {output_file}")
        print("Use --force to regenerate.")
        return
    
    # Load original data
    npz_file = dataset_dir / f"{dataset_name}_{split}_{target_size}x{target_size}.npz"
    
    if not npz_file.exists():
        raise FileNotFoundError(f"{dataset_name} {split} data not found at {npz_file}")
    
    print(f"Loading {split} data from {npz_file}...")
    images, labels = load_dataset_npz(npz_file, target_size)
    
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
    
    # Check if classes are balanced
    counts_list = list(Original.values())
    is_balanced = len(set(counts_list)) == 1
    
    # PREPROCESS: Calculate balancing parameters
    if is_balanced:
        print(f"\n  ✓ Classes are balanced ({counts_list[0]:,} per class)")
        print(f"\nCalculating balancing parameters (simple augment_ratio={augment_ratio})...")
        # Simple case: Percent array where each entry is the same (10% for all classes)
        Percent = {i: augment_ratio for i in range(10)}
    else:
        print(f"\n  ⚠ Classes are not balanced (min: {min(counts_list):,}, max: {max(counts_list):,})")
        print(f"\nCalculating balancing parameters (using balancing algorithm)...")
        Percent, Augmented, m = calculate_balancing_parameters(Original)
        print(f"\nBalancing plan:")
        for i in range(10):
            print(f"  Class {i}: {Original[i]:,} original → {Augmented[i]:,} augmented → {Original[i] + Augmented[i]:,} final")
        print(f"  Target per class: {m:,}")
    
    # Organize images by class
    print(f"\nOrganizing images by class...")
    class_images = {i: [] for i in range(10)}
    for img, label in zip(images, labels):
        class_images[label].append(img)
    
    # Generate augmented data
    print(f"\nGenerating augmented images...")
    augmented_images = []
    augmented_labels = []
    
    for class_idx in range(10):
        num_to_augment = int(math.floor(Original[class_idx] * Percent[class_idx]))
        class_img_list = class_images[class_idx]
        
        # Check if we have enough images to augment
        if num_to_augment > len(class_img_list):
            raise ValueError(f"Class {class_idx}: Cannot augment {num_to_augment} images, only {len(class_img_list)} available")
        
        # Randomly select images to augment
        indices_to_augment = random.sample(range(len(class_img_list)), num_to_augment)
        
        # Generate augmented versions - collect all first
        # Note: augment_image returns 6 images (original + 5 augmented), 
        # but we only want the 5 augmented ones
        all_aug_versions = []
        for idx in indices_to_augment:
            img = class_img_list[idx]
            aug_versions = augment_image(img, class_idx)
            # Skip the first one (original) - we only want the 5 augmented versions
            all_aug_versions.extend(aug_versions[1:])
        
        # POSTPROCESS: Randomly select exactly Augmented[class_idx] from all generated versions
        if is_balanced:
            selected = all_aug_versions
        else:
            if Augmented[class_idx] > len(all_aug_versions):
                raise ValueError(f"Class {class_idx}: Cannot select {Augmented[class_idx]} augmented images, only {len(all_aug_versions)} generated")
            selected = random.sample(all_aug_versions, Augmented[class_idx])
        
        if selected:
            imgs, aug_labels = zip(*selected)
            augmented_images.extend(imgs)
            augmented_labels.extend(list(aug_labels))  # Convert tuple to list before extending
        
        print(f"  Class {class_idx}: {len(selected):,} augmented images created (from {len(all_aug_versions):,} generated)")
    
    # Combine original and augmented images (no post-processing yet)
    print(f"\nCombining original and augmented images...")
    print(f"  Original: {len(images):,} images, labels shape: {labels.shape if hasattr(labels, 'shape') else 'N/A'}, len: {len(labels):,}")
    print(f"  Augmented: {len(augmented_images):,} images, {len(augmented_labels):,} labels")
    
    # Convert to lists, ensuring labels is 1D
    if isinstance(labels, np.ndarray):
        if labels.ndim > 1:
            labels = labels.flatten()
        all_labels = labels.tolist()
    else:
        all_labels = list(labels)
    
    all_images = list(images)
    all_images.extend(augmented_images)
    all_labels.extend(augmented_labels)
    
    # Apply post-processing to all final images
    print(f"\nApplying post-processing to final images...")
    processed_images = []
    for img in all_images:
        processed = apply_post_processing(img)
        processed_images.append(processed)
    
    all_images = np.array(processed_images)
    all_labels = np.array(all_labels, dtype=np.int32)
    
    # Verify counts match
    print(f"\nTotal images: {len(all_images):,}, Total labels: {len(all_labels):,}")
    if len(all_images) != len(all_labels):
        raise ValueError(f"Mismatch: {len(all_images):,} images but {len(all_labels):,} labels")
        
    # Verify final distribution
    final_counts = Counter(all_labels.tolist() if isinstance(all_labels, np.ndarray) else all_labels)
    print(f"\nFinal class distribution:")
    for i in range(10):
        count = final_counts.get(i, 0)
        print(f"  Class {i}: {count:,}")
    
    # Verify total matches
    total_counted = sum(final_counts.values())
    print(f"\nTotal counted: {total_counted:,}, Expected: {len(all_labels):,}")
    
    # Save
    print(f"\nSaving to {output_file}...")
    # Convert to uint8 for smaller file size
    x_uint8 = (all_images * 255).astype(np.uint8)
    np.savez_compressed(output_file, x=x_uint8, y=all_labels)
    
    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"  Saved {len(all_images):,} images")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  ✓ Complete!")
