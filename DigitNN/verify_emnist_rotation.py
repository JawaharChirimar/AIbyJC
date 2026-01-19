#!/usr/bin/env python3
"""
Verify EMNIST digits are correctly oriented after rotation fix
"""

import numpy as np
import matplotlib.pyplot as plt
from GetEMNIST import load_emnist_from_zip

# Load EMNIST Digits from zip (with rotation fix applied)
print("Loading EMNIST Digits from zip file (with rotation fix)...")
x_train, y_train, x_test, y_test = load_emnist_from_zip(split='digits')

if x_train is None:
    print("Failed to load EMNIST Digits")
    exit(1)

print(f"Loaded {len(x_train):,} training samples")
print(f"Image shape: {x_train[0].shape}")
print(f"Data type: {x_train.dtype}")
print(f"Value range: {x_train.min()} to {x_train.max()}")

# Select 10 random samples, one for each digit 0-9
np.random.seed(42)
selected_indices = []
for digit in range(10):
    # Find first occurrence of each digit
    digit_indices = np.where(y_train == digit)[0]
    if len(digit_indices) > 0:
        idx = np.random.choice(digit_indices)
        selected_indices.append((idx, digit))

# Create figure with 10 subplots
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for i, (idx, label) in enumerate(selected_indices):
    img = x_train[idx]
    
    # Display the image
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Digit: {label}\nIndex: {idx}')
    axes[i].axis('off')

plt.suptitle('10 EMNIST Digits (After Rotation Fix)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Also print info
print("\n10 EMNIST Digits (one per digit 0-9):")
for idx, label in selected_indices:
    print(f"  Digit {label}: index {idx}")
