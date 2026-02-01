#!/usr/bin/env python3
"""
StratifiedBatchGenerator.py

Creates stratified batch generators that ensure balanced class representation
in each batch.

For digit classification:
- Each batch has equal samples from each digit class (0-9)
- Non-digits (class 10) appear proportionally (14.6% of batch)
- Batch size is exactly maintained (removes 1 random sample if needed)
"""

import numpy as np


def create_stratified_batch_generator(x_data, y_data, batch_size=128, num_classes=11, non_digit_class=10):
    """
    Create a generator that yields stratified batches with balanced class representation.
    
    Each batch contains:
    - Equal samples from each digit class (0-9)
    - Proportional non-digits (class 10) based on dataset ratio
    - Exact batch_size (removes 1 random sample if needed)
    
    Args:
        x_data: numpy array of shape (N, H, W, C) - input images
        y_data: numpy array of shape (N,) - class labels (0-10)
        batch_size: Target batch size (default 128)
        num_classes: Total number of classes (default 11: 0-9 digits + 10 non-digits)
        non_digit_class: Class label for non-digits (default 10)
    
    Returns:
        Generator function that yields (batch_x, batch_y) tuples
    """
    n_samples = len(x_data)
    
    # Separate indices by class
    class_indices = {}
    for class_idx in range(num_classes):
        class_indices[class_idx] = np.where(y_data == class_idx)[0]
        print(f"  Class {class_idx}: {len(class_indices[class_idx]):,} samples")
    
    # Calculate samples per batch dynamically based on num_classes and batch_size
    # Example: 10 classes, batch_size=64 → 64/10 = 6.4 → 4 classes get 7, 6 classes get 6
    # Example: 11 classes, batch_size=128 → 128/11 = 11.636... → 7 classes get 12, 4 classes get 11
    samples_per_class_base = batch_size // num_classes  # Floor division
    remainder = batch_size % num_classes  # How many extra samples to distribute
    
    samples_per_class_low = samples_per_class_base
    samples_per_class_high = samples_per_class_base + 1
    num_classes_high = remainder  # Classes that get the extra sample
    num_classes_low = num_classes - remainder  # Classes that get base count
    
    # Verify calculation
    total = num_classes_high * samples_per_class_high + num_classes_low * samples_per_class_low
    assert total == batch_size, f"Batch size calculation error: {total} != {batch_size}"
    
    print(f"\nStratified batch composition:")
    if num_classes_high > 0:
        print(f"  {num_classes_high} classes: {samples_per_class_high} samples each")
    if num_classes_low > 0:
        print(f"  {num_classes_low} classes: {samples_per_class_low} samples each")
    print(f"  Total: {total} (exact batch_size)")
    print(f"  Selection of which classes get {samples_per_class_high} vs {samples_per_class_low} is random per batch")
    
    # Find minimum class count to ensure true balance
    min_class_count = min(len(class_indices[class_idx]) for class_idx in range(num_classes))
    balanced_samples_per_epoch = min_class_count * num_classes
    
    print(f"\nBalanced stratified sampling:")
    print(f"  Minimum class count: {min_class_count:,}")
    print(f"  Samples per epoch: {balanced_samples_per_epoch:,} ({min_class_count:,} per class)")
    print(f"  Batches per epoch: {balanced_samples_per_epoch // batch_size}")
    
    def stratified_generator():
        """Generator that yields stratified batches with true balance."""
        
        while True:
            # At start of each epoch, randomly sample min_class_count from each class
            epoch_indices = {}
            for class_idx in range(num_classes):
                # Randomly sample min_class_count indices from this class
                all_class_indices = class_indices[class_idx]
                sampled_indices = np.random.choice(all_class_indices, size=min_class_count, replace=False)
                # Shuffle the sampled indices for this epoch
                np.random.shuffle(sampled_indices)
                epoch_indices[class_idx] = sampled_indices
            
            # Calculate batches for this epoch
            batches_per_epoch = balanced_samples_per_epoch // batch_size
            class_positions = {i: 0 for i in range(num_classes)}
            
            # Generate batches for this epoch
            for _ in range(batches_per_epoch):
                batch_indices = []
                
                # Randomly select which classes get the higher sample count (rest get lower)
                all_classes = list(range(num_classes))
                np.random.shuffle(all_classes)
                classes_high = all_classes[:num_classes_high]  # Classes that get samples_per_class_high
                classes_low = all_classes[num_classes_high:]    # Classes that get samples_per_class_low
                
                # Sample from each class's epoch indices
                for class_idx in range(num_classes):
                    indices = epoch_indices[class_idx]
                    pos = class_positions[class_idx]
                    
                    # Determine how many samples this class gets
                    if class_idx in classes_high:
                        samples_needed = samples_per_class_high
                    else:
                        samples_needed = samples_per_class_low
                    
                    # Take samples (cycle if needed)
                    end_pos = pos + samples_needed
                    if end_pos <= len(indices):
                        batch_indices.extend(indices[pos:end_pos])
                        class_positions[class_idx] = end_pos
                    else:
                        # Wrap around if we exceed (shouldn't happen with balanced sampling, but handle gracefully)
                        batch_indices.extend(indices[pos:])
                        remaining = end_pos - len(indices)
                        batch_indices.extend(indices[:remaining])
                        class_positions[class_idx] = remaining
                
                # Shuffle the batch to mix classes
                np.random.shuffle(batch_indices)
                
                # Yield batch
                yield x_data[batch_indices], y_data[batch_indices]
    
    return stratified_generator


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing StratifiedBatchGenerator...")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 10000
    n_digits_per_class = 900  # 900 * 10 = 9000 digits
    n_non_digits = 1000  # 1000 non-digits
    
    # Create labels
    y_data = []
    for digit in range(10):
        y_data.extend([digit] * n_digits_per_class)
    y_data.extend([10] * n_non_digits)
    y_data = np.array(y_data, dtype=np.int32)
    
    # Shuffle labels
    perm = np.random.permutation(len(y_data))
    y_data = y_data[perm]
    
    # Create dummy images
    x_data = np.random.rand(n_samples, 64, 64, 1).astype(np.float32)
    
    print(f"Test data: {n_samples} samples")
    print(f"  Digits (0-9): {n_digits_per_class} each")
    print(f"  Non-digits (10): {n_non_digits}")
    print(f"  Non-digit ratio: {n_non_digits/n_samples*100:.1f}%")
    print()
    
    # Create generator
    batch_size = 128
    gen_func = create_stratified_batch_generator(x_data, y_data, batch_size=batch_size)
    gen = gen_func()  # Call the function to get the generator
    
    # Test a few batches
    print("Testing batch composition...")
    print("-" * 60)
    
    for batch_num in range(5):
        batch_x, batch_y = next(gen)
        
        # Count classes in batch
        unique, counts = np.unique(batch_y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        print(f"Batch {batch_num + 1}:")
        print(f"  Size: {len(batch_y)}")
        print(f"  All classes (0-10): ", end="")
        all_counts = [class_counts.get(i, 0) for i in range(11)]
        print(f"{all_counts}")
        print(f"  Classes with 12: {[i for i, c in enumerate(all_counts) if c == 12]}")
        print(f"  Classes with 11: {[i for i, c in enumerate(all_counts) if c == 11]}")
        print(f"  Min: {min(all_counts)}, Max: {max(all_counts)}")
        print()
    
    print("=" * 60)
    print("Test complete!")
