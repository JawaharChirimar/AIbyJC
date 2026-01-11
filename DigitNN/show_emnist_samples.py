#!/usr/bin/env python3
"""
Show 10 random EMNIST digit samples before transformation
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available, will print image arrays instead")

from emnist import extract_training_samples

# Load EMNIST Digits
print("Loading EMNIST Digits...")
x_train, y_train = extract_training_samples('digits')
print(f"Loaded {len(x_train)} samples")
print(f"Image shape: {x_train[0].shape}")
print(f"Data type: {x_train.dtype}")
print(f"Value range: {x_train.min()} to {x_train.max()}")

# Select 10 random samples
np.random.seed(42)  # For reproducibility
random_indices = np.random.choice(len(x_train), 10, replace=False)

if HAS_MATPLOTLIB:
    # Create figure with 10 subplots
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        img = x_train[idx]
        label = y_train[idx]
        
        # Display the image
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Sample {i+1}\nDigit: {label}\nShape: {img.shape}')
        axes[i].axis('off')

    plt.suptitle('10 Random EMNIST Digits (BEFORE transformation)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('/Users/jawaharchirimar/Python/emnist_samples_before.png', dpi=150, bbox_inches='tight')
    print("Saved image to: emnist_samples_before.png")
    plt.show()
else:
    # Print ASCII representation if matplotlib not available
    print("\n10 Random EMNIST Digits (BEFORE transformation):")
    print("=" * 60)
    for i, idx in enumerate(random_indices):
        img = x_train[idx]
        label = y_train[idx]
        print(f"\nSample {i+1}: Digit {label}")
        print(f"Shape: {img.shape}")
        # Print a simple ASCII representation
        threshold = img.max() / 2
        for row in img[:14]:  # Show first 14 rows
            line = ''.join(['#' if pixel > threshold else '.' for pixel in row[:14]])
            print(line)
        print("...")

