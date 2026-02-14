#!/usr/bin/env python3
"""
PregenAugmentedDataMNIST.py

Loads pre-generated augmented training data from separate dataset files.
Each dataset has its own augmented file (EMNIST, ARDIS, USPS, Google Fonts, Non-digits).

This replaces the old approach of a single combined augmented file.
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from collections import Counter
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from DataManagement.PregenAugmentedBase import (
    load_dataset_data,
    apply_post_processing,
    augment_image_by_index,
    DATA_DIR,
    NUMBER_OF_AUGMENTATIONS
)

def str2bool(value):
    """Parse common CLI boolean strings into a real bool."""
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("true", "t", "1", "yes", "y"):
        return True
    if lowered in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_dataset_from_table(train_tableX, train_x, train_y, 
    split, image_size):
    for key in train_tableX:
        dataset_dir = DATA_DIR / train_tableX[key]["input_path"]
        suffix = train_tableX[key].get("suffix", "")
        images, labels = load_dataset_data(train_tableX[key]["name"], 
        dataset_dir, split=split, target_size=image_size, suffix=suffix)
        train_x[key] = images
        train_y[key] = labels
    

def get_original_counts_by_class(y_data):
    original_counts = Counter(y_data)
    original_counts_split = {i: original_counts.get(i, 0) for i in range(10)}
    
    return original_counts_split


def augment_count_for_class(OriginalCounts, augment_ratio):
    """
    Calculate the number of augmentations for each class.
    Args:
        OriginalCounts: Original counts of each digit. 
        Array of 10 integers.
        augment_ratio: Fraction of the class with max samples to be augmented. 
        Default is 50%. 0 < augment_ratio < 1.
    Returns:
        AugmentCounts: Number of augmentations for each digit.
        Array of 10 integers.
    """

    if augment_ratio <= 0.0 or augment_ratio >= 1.0:
        raise ValueError(f"Augment ratio must be between 0 and 1, got {augment_ratio}")

    m = max(OriginalCounts.values())
    max_augment_count = int(math.floor(m * augment_ratio))
    max_class_count = m + max_augment_count

    AugmentCounts = [0 for i in range(10)]
    for i in range(10):
        AugmentCounts[i] = max_class_count - OriginalCounts[i]
        if AugmentCounts[i] < 0:
            raise ValueError(f"Class {i}: {AugmentCounts[i]}. Augment count cannot be negative")

    return AugmentCounts


def process_dataset_data(output_path, dataset_name, 
images, labels, 
OriginalCounts, 
split, target_size, force, augment_ratio):
    """
    Process one dataset split with class-wise augmentation and save it.
    
    Args:
        output_path: Path to output directory (e.g., "augmented", "EMNIST", "ardis", "usps")
        dataset_name: Name prefix for files (e.g., "mnist_usps_ardis_customone" or "mnist")
        images: List of images
        labels: List of labels
        OriginalCounts: Original counts of each digit
        split: 'train' or 'test'
        target_size: Image size (28 or 64)
        force: Force regeneration if file exists
        augment_ratio: Fraction used to compute class-wise augmentation targets
    """
    print(f"\n{'='*70}")
    print(f"Processing Combined {split.upper()} data ({target_size}x{target_size})")
    print(f"{'='*70}")
    
    # Output file
    output_file = DATA_DIR / output_path / f"{dataset_name}_{split}_augmented_{target_size}x{target_size}.npz"
    
    if output_file.exists() and not force:
        print(f"Augmented data already exists at {output_file}")
        print("Use --force to regenerate.")
        return
    
    if images is None or labels is None:
        raise ValueError(f"Empty data for {dataset_name} {split}!")
    if len(images) == 0 or len(labels) == 0:
        raise ValueError(f"No samples found for {dataset_name} {split}!")
    if len(images) != len(labels):
        raise ValueError(
            f"Image/label length mismatch for {dataset_name} {split}: "
            f"{len(images):,} images vs {len(labels):,} labels"
        )
    
    print(f"  Given {len(images):,} samples")
    print(f"  Labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}, len: {len(labels):,}")
        
    # Count original distribution
    print(f"\nOriginal class distribution:")
    for i in range(10):
        if OriginalCounts[i] < 1:
            raise ValueError(f"Class {i}: {OriginalCounts[i]}. Original count cannot be 0 or less")
        print(f"  Class {i}: {OriginalCounts[i]:,}")
    
    # Organize images by class
    print(f"\nOrganizing images by class...")
    class_images = {i: [] for i in range(10)}
    for sample_idx, (img, label) in enumerate(zip(images, labels)):
        if label not in class_images:
            raise ValueError(
                f"Invalid label at index {sample_idx} for {dataset_name} {split}: "
                f"label={label!r}. Expected integer class in [0, 9]."
            )
        class_images[label].append(img)
    
    # Generate augmented data
    print(f"\nGenerating augmented images...")
    augmented_images = []
    augmented_labels = []
        
    AugmentCounts = augment_count_for_class(OriginalCounts, augment_ratio)

    for class_idx in range(10):
        print(f"  Class {class_idx}: {OriginalCounts[class_idx]:,} original → {AugmentCounts[class_idx]:,} augmented → {OriginalCounts[class_idx] + AugmentCounts[class_idx]:,} final")
        num_to_augment = AugmentCounts[class_idx]
        class_img_list = class_images[class_idx]
                
        # Randomly select images to augment
        if num_to_augment == 0:
            indices_to_augment = []
        elif num_to_augment > len(class_img_list):
            print(f"Class {class_idx}: Augmented images needed ({num_to_augment}) >  Original images ({len(class_img_list)})")
            print(f"Class {class_idx}: Choosing indices with replacement to enable this.")
            indices_to_augment = np.random.choice(range(len(class_img_list)), num_to_augment, replace=True)    
        else:
            print(f"Class {class_idx}: Augmented images needed ({num_to_augment}) <=  Original images ({len(class_img_list)})")
            print(f"Class {class_idx}: Choosing indices without replacement to enable this.")
            indices_to_augment = np.random.choice(range(len(class_img_list)), num_to_augment, replace=False)
            

        # Generate augmented versions - collect all first
        # Note: augment_image_by_index returns 1 augmented image
        all_aug_versions = []
        for idx in indices_to_augment:
            img = class_img_list[idx]
            distortion_index = np.random.randint(0, NUMBER_OF_AUGMENTATIONS)
            aug_version = augment_image_by_index(img, class_idx, distortion_index)
            if aug_version is None:
                raise ValueError(f"Augmentation index {distortion_index} returned None for class {class_idx}")
            all_aug_versions.append(aug_version)
        
        if all_aug_versions:
            imgs, aug_labels = zip(*all_aug_versions)
            augmented_images.extend(imgs)
            augmented_labels.extend(list(aug_labels))  # Convert tuple to list before extending
        
        print(f"  Class {class_idx}: {len(all_aug_versions):,} augmented versions generated")
    
    # Combine original and augmented images (no post-processing yet)
    print(f"\nCombining original and augmented images...")
    print(f"  Original: {len(images):,} images, labels shape: {labels.shape if hasattr(labels, 'shape') else 'N/A'}, len: {len(labels):,}")
    print(f"  Augmented: {len(augmented_images):,} images, {len(augmented_labels):,} labels")
    
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
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Convert to uint8 for smaller file size
    all_images = np.asarray(all_images)
    if all_images.dtype == np.uint8:
        x_uint8 = all_images
    else:
        x_uint8 = np.clip(all_images, 0.0, 1.0)
        x_uint8 = (x_uint8 * 255).round().astype(np.uint8)
    np.savez_compressed(output_file, x=x_uint8, y_softmax=all_labels)
    
    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"  Saved {len(all_images):,} images")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  ✓ Complete!")


def create_augmented_data(augment_ratio, image_size, force, datasets):
    """
    Create augmented training and test datasets from separate files.
    
    Args:
        augment_ratio: Fraction of each class to augment
        image_size: Image size (28 or 64)
        force: Force regeneration if file exists
        datasets: 
        'mnist_plus': MNIST + USPS + ARDIS 
        'mnist': MNIST only
        'emnist': EMNIST only
        'ardis': ARDIS only
        'usps': USPS only
        Note: google_fonts is not included here because it 
        is augmented in PregenGoogleFonts.py
        Note: non_digits is not included here because it 
        is augmented in PregenNonDigits.py
    """
    
    print(f"Creating augmented training data for {datasets} ({image_size}x{image_size})...")
    
    # =========================================================================
    # For training data
    # =========================================================================
    # Note: "name" must match the file prefix (e.g., "mnist" for mnist_train_64x64.npz)
    #'mnist_plus', 'mnist', 'emnist', 'ardis', 'usps'
    train_table = {}
    if datasets == 'mnist_plus':
        train_table = {
            "mmnist": {"name": "mnist", "input_path": "MNIST"},
            "ardis": {"name": "ardis", "input_path": "ardis"},
            "usps": {"name": "usps", "input_path": "usps"},
        }
        output_path = "augmented"
        dataset_name = "mnist_usps_ardis_customone"
    elif datasets == 'mnist':
        train_table = {
            "mmnist": {"name": "mnist", "input_path": "MNIST"},
        }
        output_path = "augmented"
        dataset_name = "mnist"
    elif datasets == 'emnist':
        train_table = {
            "emnist": {"name": "emnist_digits", "input_path": "EMNIST"},
        }
        output_path = "EMNIST"
        dataset_name = "emnist"
    elif datasets == 'ardis':
        train_table = {
            "ardis": {"name": "ardis", "input_path": "ardis"},
        }
        output_path = "ardis"
        dataset_name = "ardis"
    elif datasets == 'usps':
        train_table = {
            "usps": {"name": "usps", "input_path": "usps"},
        }
        output_path = "usps"
        dataset_name = "usps"
    else:
        raise ValueError(f"Invalid dataset: {datasets}")

    train_x = {}
    train_y = {}
    load_dataset_from_table(train_table, train_x, train_y, 'train', image_size)

    test_x = {}
    test_y = {}
    load_dataset_from_table(train_table, test_x, test_y, 'test', image_size)

    if len(train_x) == 0 or len(train_y) == 0:
        raise ValueError("No training datasets found!")
    
    if len(test_x) == 0 or len(test_y) == 0:
        raise ValueError("No testing datasets found!")
    
    # Concatenate across all dictionary keys
    x_train = np.concatenate(list(train_x.values()), axis=0)
    y_train = np.concatenate(list(train_y.values()), axis=0)
    x_test = np.concatenate(list(test_x.values()), axis=0)
    y_test = np.concatenate(list(test_y.values()), axis=0)

    print('='*70)
    print(f"\nTotal training samples: {len(x_train):,}")
    print("Training datasets:")
    for key in train_x:
        print(f"  - {train_table[key]['name']}: {len(train_x[key]):,}")

    print(f"\nTotal testing samples: {len(x_test):,}")
    print("Testing datasets:")
    for key in test_x:
        print(f"  - {train_table[key]['name']}: {len(test_x[key]):,}")
    print('='*70)

    OriginalCountsTrain = get_original_counts_by_class(y_train)
    OriginalCountsTest = get_original_counts_by_class(y_test)

    np.random.seed(41)
    process_dataset_data(output_path, dataset_name, 
    x_train, y_train, 
    OriginalCountsTrain, 
    'train', image_size, force, augment_ratio)

    np.random.seed(97)
    process_dataset_data(output_path, dataset_name, 
    x_test, y_test, 
    OriginalCountsTest, 
    'test', image_size, force, augment_ratio)


def main():
    parser = argparse.ArgumentParser(description="Create augmented training and test data for specified datasets")
    parser.add_argument("--size", type=int, required=True, choices=[28, 64],
                        help="Image size (28 or 64)")
    parser.add_argument("--augment_ratio", type=int, required=True, choices=range(1, 56),
                        help="Augmentation ratio. Enter an integer in [1, 55].")
    parser.add_argument("--force", type=str2bool, required=True,
                        help="Force regeneration even if file exists (true/false)")
    parser.add_argument("--datasets", type=str, required=True,
    choices=['mnist_plus', 'mnist', 'emnist', 'ardis', 'usps'], 
    help="Create augmented data for the specified datasets. Choices: mnist_plus, mnist, emnist, ardis, usps")
    args = parser.parse_args()
    
    # Convert percentage to decimal (10 -> 0.10)
    augment_ratio = args.augment_ratio / 100.0
    
    print(f"\n{'='*70}")
    print(f"Running with:")
    print(f"  Image size={args.size}")
    
    if args.force:
        print(f"  Force regeneration of npz files for train and test data")
    else:
        print(f"  Generate data only if npz files do not exist for train and/or test") 
    
    print(f"  Augmentation ratio={args.augment_ratio}% ({augment_ratio})")
    
    if args.datasets == 'mnist_plus':
        print(f"  Creating augmented data for MNIST + USPS + ARDIS")
    elif args.datasets == 'mnist':
        print(f"  Creating augmented data for MNIST only")
    elif args.datasets == 'emnist':
        print(f"  Creating augmented data for EMNIST only")
    elif args.datasets == 'ardis':
        print(f"  Creating augmented data for ARDIS only")
    elif args.datasets == 'usps':
        print(f"  Creating augmented data for USPS only")
    else:
        print(f"  Invalid dataset: {args.datasets}")
        print(f"  Choices: mnist_plus, mnist, emnist, ardis, usps")
        raise ValueError(f"Invalid dataset: {args.datasets}")

    print(f"{'='*70}\n")
    
    create_augmented_data(
        augment_ratio=augment_ratio,
        image_size=args.size,
        force=args.force,
        datasets=args.datasets
    )


if __name__ == "__main__":
    main()
