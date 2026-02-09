#!/usr/bin/env python3
"""
PregenAugmentedData.py

Loads pre-generated augmented training data from separate dataset files.
Each dataset has its own augmented file (EMNIST, ARDIS, USPS, Google Fonts, Non-digits).

This replaces the old approach of a single combined augmented file.
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from collections import Counter
import random
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from DataManagement.PregenAugmentedBase import (
    load_dataset_data,
    apply_post_processing,
    augment_image,
    DATA_DIR,
    AUGMENT_RATIO
)

def calculate_balancing_parameters(Original, augment_ratio_for_max_class):
    """
    Calculate augmentation percentages and counts using balancing algorithm.
    
    Args:
        Original: dict of {class_idx: count} for original class distribution
    
    Returns:
        PreAugmented: dict of {class_idx: num_augmented_images_to_create}
        Residue: dict of {class_idx: num_augmented_images_to_create}
        m: target count per class after balancing
    """
    import math
    
    Percent = {}
    PreAugmented = {}
    Residue = {}

    maxIndex = max(Original, key=Original.get)    
    # Step 1: Calculate Percent[maxIndex] and m
    Percent[maxIndex] = augment_ratio_for_max_class
    PreAugmented[maxIndex] = int(math.floor(Original[maxIndex] * Percent[maxIndex]))
    m = Original[maxIndex] + PreAugmented[maxIndex] * 5# Initialize maximum
    Residue[maxIndex] = 0  # No residue for max class
    
    # Step 2: Calculate Percent[k] for k = 1 to 9, and track minimum
    for k in range(0, 10):
        if k == maxIndex:
            continue
        currentAugmented = (m - Original[k]) / 5.0
        currentAugmentedFloor = int(math.floor(currentAugmented))
        if currentAugmented > currentAugmentedFloor:
            PreAugmented[k] = currentAugmentedFloor + 1
            Residue[k] = PreAugmented[k]*5 - (m - Original[k])
        else:
            PreAugmented[k] = currentAugmentedFloor
            Residue[k] = 0
    
    return PreAugmented, Residue, m

def adjust_custom_one_and_mnist(train_x, train_y, test_x, test_y):
    """
    Validate CustomOne and MNIST counts, then swap 1000 MNIST 1's with CustomOne.
    Modifies train_x, train_y, test_x, test_y dictionaries in place.
    
    Returns:
        True if all checks pass and swap completed, False otherwise.
    """
    MNIST_TRAIN_CLASS1_NUM = 6742
    MNIST_TEST_CLASS1_NUM = 1135
    CUSTOM_ONE_TRAIN_NUM = 1000
    CUSTOM_ONE_TEST_NUM = 250
    
    if len(train_x["custom_one"]) != CUSTOM_ONE_TRAIN_NUM:
        print(f"\nERROR: Custom ONE training dataset has {len(train_x['custom_one'])} samples. Expected {CUSTOM_ONE_TRAIN_NUM}!")
        return False
    
    if len(test_x["custom_one"]) != CUSTOM_ONE_TEST_NUM:
        print(f"\nERROR: Custom ONE testing dataset has {len(test_x['custom_one'])} samples. Expected {CUSTOM_ONE_TEST_NUM}!")
        return False

    # Check count of 1's in MNIST train
    mnist_train_class1_count = np.sum(train_y["mmnist"] == 1)
    if mnist_train_class1_count != MNIST_TRAIN_CLASS1_NUM:
        print(f"\nERROR: MNIST train has {mnist_train_class1_count} class 1 samples. Expected {MNIST_TRAIN_CLASS1_NUM}!")
        return False
    
    # Check count of 1's in MNIST test
    mnist_test_class1_count = np.sum(test_y["mmnist"] == 1)
    if mnist_test_class1_count != MNIST_TEST_CLASS1_NUM:
        print(f"\nERROR: MNIST test has {mnist_test_class1_count} class 1 samples. Expected {MNIST_TEST_CLASS1_NUM}!")
        return False
    
    print(f"\n{'='*70}")
    print("Swapping MNIST 1's with CustomOne...")
    print(f"{'='*70}")
    
    # Remove 1000 random 1's from MNIST train data
    mnist_train_class1_indices = np.where(train_y["mmnist"] == 1)[0]
    indices_to_remove = np.random.choice(mnist_train_class1_indices, size=CUSTOM_ONE_TRAIN_NUM, replace=False)
    mask = np.ones(len(train_y["mmnist"]), dtype=bool)
    mask[indices_to_remove] = False
    train_x["mmnist"] = train_x["mmnist"][mask]
    train_y["mmnist"] = train_y["mmnist"][mask]
    print(f"  Removed {CUSTOM_ONE_TRAIN_NUM} random class 1 samples from MNIST train")
    
    # Remove 250 random 1's from MNIST test data
    mnist_test_class1_indices = np.where(test_y["mmnist"] == 1)[0]
    indices_to_remove = np.random.choice(mnist_test_class1_indices, size=CUSTOM_ONE_TEST_NUM, replace=False)
    mask = np.ones(len(test_y["mmnist"]), dtype=bool)
    mask[indices_to_remove] = False
    test_x["mmnist"] = test_x["mmnist"][mask]
    test_y["mmnist"] = test_y["mmnist"][mask]
    print(f"  Removed {CUSTOM_ONE_TEST_NUM} random class 1 samples from MNIST test")
    
    # Verify final counts
    final_mnist_train_class1 = np.sum(train_y["mmnist"] == 1)
    final_mnist_test_class1 = np.sum(test_y["mmnist"] == 1)
    print(f"  MNIST train class 1: {MNIST_TRAIN_CLASS1_NUM} → {final_mnist_train_class1}")
    print(f"  MNIST test class 1: {MNIST_TEST_CLASS1_NUM} → {final_mnist_test_class1}")
    print(f"  CustomOne train: {len(train_x['custom_one'])} (all class 1)")
    print(f"  CustomOne test: {len(test_x['custom_one'])} (all class 1)")
    print(f"{'='*70}")
    
    return True

def load_dataset_from_table(train_tableX, train_x, train_y, split):
    for key in train_tableX:
        dataset_dir = DATA_DIR / train_tableX[key]["path1"]
        suffix = train_tableX[key].get("suffix", "")
        images, labels = load_dataset_data(train_tableX[key]["name"], dataset_dir, split=split, target_size=64, suffix=suffix)
        train_x[key] = images
        train_y[key] = labels
    

def balance_datasets(y_data, data_name, augment_ratio):
   # OriginalCountsTrain, PreAugmented, Residue, m = balance_datasets(y_train)

    original_counts = Counter(y_data)
    original_counts_split = {i: original_counts.get(i, 0) for i in range(10)}
    print(f"\nOriginal class distribution for {data_name} data:")
    for i in range(10):
        print(f"  Class {i}: {original_counts_split[i]:,}")
        
    # Check if classes are balanced
    counts_list = list(original_counts_split.values())
    is_balanced = len(set(counts_list)) == 1
    
    # PREPROCESS: Calculate balancing parameters
    if is_balanced:
        print(f"\n  ✓ Classes are balanced ({counts_list[0]:,} per class)")
        # Simple case: Percent array where each entry is the same (10% for all classes)
        Residue = {i: 0 for i in range(10)}
        xstep = math.floor(original_counts_split[0] * augment_ratio)
        PreAugmented = {i: xstep for i in range(10)}
        m = xstep + original_counts_split[0]
        for i in range(10):
            print(f"  Class {i}: {original_counts_split[i]:,} original → {PreAugmented[i]:,} augmented → {original_counts_split[i] + 5*PreAugmented[i]:,} final")
        print(f"  Target per class: {original_counts_split[0] * augment_ratio   :,}")
    else:
        print(f"\n  ⚠ Classes are not balanced (min: {min(counts_list):,}, max: {max(counts_list):,})")
        print(f"\nCalculating balancing parameters (using balancing algorithm)...")
        PreAugmented, Residue, m = calculate_balancing_parameters(original_counts_split, augment_ratio)
        print(f"\nBalancing plan:")
        for i in range(10):
            print(f"  Class {i}: {original_counts_split[i]:,} original → {PreAugmented[i]:,} augmented → {-Residue[i]:,} residue → {original_counts_split[i] + 5*PreAugmented[i] - Residue[i]:,} final")
        print(f"  Target per class: {m:,}")

    return original_counts_split, PreAugmented, Residue, m


def process_dataset_data(dataset_name, images, labels, 
OriginalCounts, PreAugmentedCounts, ResidueCounts, 
split, target_size, force, augment_ratio):
    """
    Generic function to process dataset with augmentation.
    Automatically detects if classes are balanced and uses appropriate method:
    - If balanced: uses simple augment_ratio (same % for all classes)
    - If not balanced: uses balancing algorithm to balance classes
    
    Args:
        dataset_name: Name prefix for files (e.g., "mnist_usps_ardis_customone" or "mnist")
        images: List of images
        labels: List of labels
        OriginalCounts: Original counts of each digit
        PreAugmentedCounts: Pre-augmented counts of each digit
        ResidueCounts: Residue counts of each digit
        split: 'train' or 'test'
        target_size: Image size (28 or 64)
        force: Force regeneration if file exists
        augment_ratio: Fraction of each class to augment (default 0.10, used if classes are balanced)
    """
    print(f"\n{'='*70}")
    print(f"Processing Combined {split.upper()} data ({target_size}x{target_size})")
    print(f"{'='*70}")
    
    # Output file
    output_file = DATA_DIR / "augmented" / f"{dataset_name}_{split}_augmented_{target_size}x{target_size}.npz"
    
    if output_file.exists() and not force:
        print(f"Augmented data already exists at {output_file}")
        print("Use --force to regenerate.")
        return
    
    if images is None or labels is None:
        raise ValueError(f"Empty data for {dataset_name} {split}!")
    
    print(f"  Given {len(images):,} samples")
    print(f"  Labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}, len: {len(labels):,}")
        
    # Count original distribution
    print(f"\nOriginal class distribution:")
    for i in range(10):
        print(f"  Class {i}: {OriginalCounts[i]:,}")
    
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
        print(f"  Class {class_idx}: {OriginalCounts[class_idx]:,} original → {PreAugmentedCounts[class_idx]:,} augmented → {-ResidueCounts[class_idx]:,} residue → {OriginalCounts[class_idx] + 5*PreAugmentedCounts[class_idx] - ResidueCounts[class_idx]:,} final")
        num_to_augment = PreAugmentedCounts[class_idx]
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
        
        if ResidueCounts[class_idx] > 0 and ResidueCounts[class_idx] < len(all_aug_versions):
            indices_to_remove = random.sample(range(len(all_aug_versions)), ResidueCounts[class_idx])
            for idx in indices_to_remove:
                all_aug_versions.pop(idx)

        selected = all_aug_versions
        if selected:
            imgs, aug_labels = zip(*selected)
            augmented_images.extend(imgs)
            augmented_labels.extend(list(aug_labels))  # Convert tuple to list before extending
        
        print(f"  Class {class_idx}: {len(selected):,} augmented images created (from {len(all_aug_versions):,} generated)")
    
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
    # Convert to uint8 for smaller file size
    x_uint8 = (all_images * 255).astype(np.uint8)
    np.savez_compressed(output_file, x=x_uint8, y_softmax=all_labels)
    
    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"  Saved {len(all_images):,} images")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  ✓ Complete!")


def load_augmented_data(augment_ratio, image_size, force, combined):
    """
    Load and combine all augmented training and test datasets from separate files.
    
    Args:
        augment_ratio: Fraction of each class to augment
        image_size: Image size (28 or 64)
        force: Force regeneration if file exists
        combined: If True, use all datasets (MNIST+USPS+ARDIS+CustomOne)
                  If False, use MNIST only
    """

    np.random.seed(42)
    
    print(f"Loading augmented training data from separate files ({image_size}x{image_size})...")
    
    # =========================================================================
    # For training data
    # =========================================================================
    # Note: "name" must match the file prefix (e.g., "mnist" for mnist_train_64x64.npz)
    if combined:
        train_table = {
            "mmnist": {"name": "mnist", "path1": "MNIST"},
            "ardis": {"name": "ardis", "path1": "ardis"},
            "usps": {"name": "usps", "path1": "usps"},
            "custom_one": {"name": "custom_one", "path1": "custom_one"},
        }
        dataset_name = "mnist_usps_ardis_customone"
    else:
        train_table = {
            "mmnist": {"name": "mnist", "path1": "MNIST"},
        }
        dataset_name = "mnist"

    train_x = {}
    train_y = {}
    load_dataset_from_table(train_table, train_x, train_y, 'train')

    test_x = {}
    test_y = {}
    load_dataset_from_table(train_table, test_x, test_y, 'test')

    if len(train_x) == 0 or len(train_y) == 0:
        raise ValueError("No training datasets found!")
    
    if len(test_x) == 0 or len(test_y) == 0:
        raise ValueError("No testing datasets found!")
    

    # Only adjust custom_one and MNIST if using combined datasets
    if combined:
        retVal = adjust_custom_one_and_mnist(train_x, train_y, test_x, test_y)
        if not retVal:
            raise ValueError("Failed to adjust custom one and mnist counts!")

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

    OriginalCountsTrain, PreAugmentedTrain, ResidueTrain, mTrain = balance_datasets(y_train, 'training', augment_ratio)
    OriginalCountsTest, PreAugmentedTest, ResidueTest, mTest = balance_datasets(y_test, 'testing', augment_ratio)

    process_dataset_data(dataset_name, 
    x_train, y_train, 
    OriginalCountsTrain, PreAugmentedTrain, ResidueTrain, 
    'train', image_size, force, augment_ratio)

    process_dataset_data(dataset_name, 
    x_test, y_test, 
    OriginalCountsTest, PreAugmentedTest, ResidueTest, 
    'test', image_size, force, augment_ratio)


def main():
    parser = argparse.ArgumentParser(description="Load and analyze MNIST + USPS + ARDIS + CustomOne datasets")
    parser.add_argument("--size", type=int, required=True, choices=[28, 64],
                        help="Image size (28 or 64)")
    parser.add_argument("--augment_ratio", type=int, default=10,
                        help="Augmentation ratio as percentage (default: 10 for 10%%)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if file exists")
    parser.add_argument("--combined", action="store_true",
                        help="Use all datasets (MNIST+USPS+ARDIS+CustomOne). If not set, use MNIST only")
    args = parser.parse_args()
    
    # Convert percentage to decimal (10 -> 0.10)
    augment_ratio = args.augment_ratio / 100.0
    
    print(f"\n{'='*70}")
    print(f"Running with:")
    print(f"  Image size={args.size}")
    if args.force:
        print(f"Force regeneration of npz files for train and test data")
    else:
        print(f"Generate data only if npz files do not exist for train and/or test") 
    print(f"Augmentation ratio={args.augment_ratio}% ({augment_ratio})")
    if args.combined:
        print(f"Using all datasets (MNIST+USPS+ARDIS+CustomOne)")
    else:
        print(f"Using MNIST only")
    print(f"{'='*70}\n")
    
    load_augmented_data(
        augment_ratio=augment_ratio,
        image_size=args.size,
        force=args.force,
        combined=args.combined
    )


if __name__ == "__main__":
    main()
