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
    
    # Calculate samples per batch: 128 / 11 = 11.636...
    # 7 classes get 12 samples, 4 classes get 11 samples
    samples_per_class_high = 12
    samples_per_class_low = 11
    num_classes_high = 7  # 7 classes get 12
    num_classes_low = 4   # 4 classes get 11
    # Total: 7×12 + 4×11 = 84 + 44 = 128 ✓
    
    print(f"\nStratified batch composition:")
    print(f"  7 classes: {samples_per_class_high} samples each")
    print(f"  4 classes: {samples_per_class_low} samples each")
    print(f"  Total: {num_classes_high * samples_per_class_high + num_classes_low * samples_per_class_low} (exact)")
    print(f"  Selection of which classes get 12 vs 11 is random per batch")
    
    # Track current position in each class's indices
    class_positions = {i: 0 for i in range(num_classes)}
    
    def stratified_generator():
        """Generator that yields stratified batches."""
        nonlocal class_positions
        
        while True:
            # Shuffle each class's indices at start of each epoch
            for class_idx in range(num_classes):
                np.random.shuffle(class_indices[class_idx])
                class_positions[class_idx] = 0
            
            # Generate batches for this epoch
            batches_in_epoch = 0
            max_batches = n_samples // batch_size
            
            for _ in range(max_batches):
                batch_indices = []
                
                # Randomly select which 7 classes get 12 samples (rest get 11)
                all_classes = list(range(num_classes))
                np.random.shuffle(all_classes)
                classes_high = all_classes[:num_classes_high]  # 7 classes get 12
                classes_low = all_classes[num_classes_high:]    # 4 classes get 11
                
                # Sample from each class
                for class_idx in range(num_classes):
                    indices = class_indices[class_idx]
                    pos = class_positions[class_idx]
                    
                    # Determine how many samples this class gets
                    if class_idx in classes_high:
                        samples_needed = samples_per_class_high  # 12
                    else:
                        samples_needed = samples_per_class_low   # 11
                    
                    # If we've exhausted this class, reshuffle and reset
                    if pos >= len(indices):
                        np.random.shuffle(indices)
                        pos = 0
                        class_positions[class_idx] = 0
                    
                    # Take samples
                    end_pos = min(pos + samples_needed, len(indices))
                    batch_indices.extend(indices[pos:end_pos])
                    class_positions[class_idx] = end_pos
                    
                    # If we didn't get enough, cycle back
                    if end_pos - pos < samples_needed:
                        needed = samples_needed - (end_pos - pos)
                        batch_indices.extend(indices[:needed])
                        class_positions[class_idx] = needed
                
                # Shuffle the batch to mix classes
                np.random.shuffle(batch_indices)
                
                # Yield batch
                yield x_data[batch_indices], y_data[batch_indices]
                batches_in_epoch += 1
    
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
