#!/usr/bin/env python3
"""
Show 10 random EMNIST digit samples
"""

import numpy as np
import matplotlib.pyplot as plt
from GetEMNIST import load_emnist_from_zip

# Load EMNIST Digits
print("Loading EMNIST Digits from zip file...")
x_train, y_train, x_test, y_test = load_emnist_from_zip(split='digits')

if x_train is None:
    print("Failed to load EMNIST Digits")
    exit(1)

print(f"Loaded {len(x_train):,} training samples")
print(f"Image shape: {x_train[0].shape}")
print(f"Data type: {x_train.dtype}")
print(f"Value range: {x_train.min()} to {x_train.max()}")

# Select 10 random samples from training set
np.random.seed(42)  # For reproducibility
random_indices = np.random.choice(len(x_train), 10, replace=False)

# Create figure with 10 subplots
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for i, idx in enumerate(random_indices):
    img = x_train[idx]
    label = y_train[idx]
    
    # Display the image
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Sample {i+1}\nDigit: {label}\nIndex: {idx}')
    axes[i].axis('off')

plt.suptitle('10 Random EMNIST Digits', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Also print some info
print("\n10 Random EMNIST Digits:")
for i, idx in enumerate(random_indices):
    print(f"  Sample {i+1}: Digit {y_train[idx]} (index {idx})")
