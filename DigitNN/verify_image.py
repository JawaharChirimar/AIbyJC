#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/jawaharchirimar/Development/AIbyJC/DigitNN')
import cv2
import numpy as np

image_path = sys.argv[1]
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

print(f"Shape: {img.shape}")
print(f"Dtype: {img.dtype}")
print(f"Min: {img.min()}, Max: {img.max()}")
unique = np.unique(img)
print(f"Unique values count: {len(unique)}")
print(f"Unique values: {unique[:20]}")

if img.shape == (28, 28):
    print("✓ 28x28")
else:
    print(f"✗ Shape: {img.shape}")

if len(img.shape) == 2:
    print("✓ 2D (grayscale)")
else:
    print(f"✗ {len(img.shape)}D")
