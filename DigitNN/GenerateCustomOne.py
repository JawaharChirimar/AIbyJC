#!/usr/bin/env python3
"""
GenerateCustomOne.py

Generates 1000 variations of a custom "1" digit style:
- Diagonal stroke from upper-left going UP-RIGHT (like /)
- Vertical stroke going DOWN from where diagonal ends
- NO horizontal base

Output: data/custom_one/ directory with NPZ files for training
"""

import numpy as np
import cv2
from pathlib import Path
import random

# Output directory
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    OUTPUT_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data" / "custom_one"
else:
    OUTPUT_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data" / "custom_one"

# Number of variations to generate
NUM_VARIATIONS = 1000

# Augmentation parameters
ROTATION_RANGE = (-10, 10)  # degrees - mild rotation
SHEAR_RANGE = (-0.15, 0.15)  # mild shearing


def draw_custom_one(size=28, stroke_width=2, diag_right=5, diag_up=3, vert_down=16, 
                    center_x=14, top_y=8):
    """
    Draw a custom "1" with simple pixel coordinates.
    
    Shape:
        /
       /
      |
      |
      |
    
    Args:
        size: Image size (28x28)
        stroke_width: Width of strokes
        diag_right: Pixels diagonal moves RIGHT (positive = right)
        diag_up: Pixels diagonal moves UP (positive = up, Y decreases)
        vert_down: Pixels vertical extends DOWN (positive = down, Y increases)
        center_x: X position where vertical is centered
        top_y: Y position where diagonal starts (top of shape)
    
    Returns:
        28x28 numpy array with white digit on black background
    """
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Diagonal start: upper-left
    diag_x1 = center_x - diag_right
    diag_y1 = top_y
    
    # Diagonal end: moved RIGHT and UP
    diag_x2 = center_x
    diag_y2 = top_y - diag_up
    
    # Vertical end: same X as diagonal end, moved DOWN
    vert_x = diag_x2
    vert_y = diag_y2 + vert_down
    
    # Clamp all coordinates to image bounds
    diag_x1 = max(1, min(size-2, diag_x1))
    diag_y1 = max(1, min(size-2, diag_y1))
    diag_x2 = max(1, min(size-2, diag_x2))
    diag_y2 = max(1, min(size-2, diag_y2))
    vert_x = max(1, min(size-2, vert_x))
    vert_y = max(1, min(size-2, vert_y))
    
    # Draw diagonal: from (diag_x1, diag_y1) to (diag_x2, diag_y2)
    cv2.line(img, (diag_x1, diag_y1), (diag_x2, diag_y2), 255, stroke_width)
    
    # Draw vertical: from (vert_x, diag_y2) to (vert_x, vert_y)
    cv2.line(img, (vert_x, diag_y2), (vert_x, vert_y), 255, stroke_width)
    
    return img


def apply_rotation(img, angle):
    """Apply mild rotation to image."""
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)
    return rotated


def apply_shear(img, shear_x):
    """Apply mild shear transformation to image."""
    h, w = img.shape
    M = np.float32([[1, shear_x, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(img, M, (w, h), borderValue=0)
    return sheared


def generate_variations():
    """Generate 1000 variations of the custom "1"."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_images = []
    all_labels = []
    
    print(f"Generating {NUM_VARIATIONS} variations of custom '1'...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    for i in range(NUM_VARIATIONS):
        # Randomize parameters with clear meaning
        stroke_width = random.choice([1, 2, 2, 2, 3])  # Mostly 2
        
        # Diagonal: moves right 4-8 pixels, up 2-5 pixels
        diag_right = random.randint(4, 5)
        diag_up = random.randint(2, 7)
        
        # Vertical: extends down 14-20 pixels
        vert_down = random.randint(18, 24)
        
        # Position: slight variation
        center_x = random.randint(12, 16)
        top_y = random.randint(6, 10)
        
        # Draw base digit
        img = draw_custom_one(
            stroke_width=stroke_width,
            diag_right=diag_right,
            diag_up=diag_up,
            vert_down=vert_down,
            center_x=center_x,
            top_y=top_y
        )
        
        # Apply mild rotation (30% chance)
        if random.random() < 0.30:
            angle = random.uniform(*ROTATION_RANGE)
            img = apply_rotation(img, angle)
        
        # Apply mild shearing (30% chance)
        if random.random() < 0.30:
            shear = random.uniform(*SHEAR_RANGE)
            img = apply_shear(img, shear)
        
        # Normalize to [0, 1]
        img_normalized = img.astype(np.float32) / 255.0
        all_images.append(img_normalized)
        all_labels.append(1)  # Label is always 1
        
        # Save individual PNG (optional, for visualization)
        if i < 40:  # Save first 20 as samples
            cv2.imwrite(str(OUTPUT_DIR / f"custom_one_{i:04d}.png"), img)
    
    # Convert to numpy arrays
    x_data = np.array(all_images)
    x_data = np.expand_dims(x_data, axis=-1)  # Shape: (N, 28, 28, 1)
    y_softmax = np.array(all_labels)  # Shape: (N,) - integer labels
    
    # Create one-hot labels for sigmoid
    y_sigmoid = np.zeros((len(all_labels), 10), dtype=np.float32)
    for i, label in enumerate(all_labels):
        y_sigmoid[i, label] = 1.0
    
    # Save NPZ files
    softmax_path = OUTPUT_DIR / "custom_one_softmax.npz"
    sigmoid_path = OUTPUT_DIR / "custom_one_sigmoid.npz"
    
    np.savez(softmax_path, x=x_data, y=y_softmax)
    np.savez(sigmoid_path, x=x_data, y=y_sigmoid)
    
    print(f"\nGenerated {len(all_images)} images")
    print(f"  Shape: {x_data.shape}")
    print(f"  Saved: {softmax_path}")
    print(f"  Saved: {sigmoid_path}")
    print(f"  Sample PNGs: {OUTPUT_DIR}/custom_one_*.png")
    
    return x_data, y_softmax


if __name__ == "__main__":
    generate_variations()
