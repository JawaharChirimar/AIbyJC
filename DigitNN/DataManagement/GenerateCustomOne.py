#!/usr/bin/env python3
"""
GenerateCustomOne.py

Generates 5000 variations of a custom "1" digit style:
- Diagonal stroke going UP and LEFT from top of vertical (like a flag)
- Vertical stroke going DOWN

Shape:
      /|
     / |
       |
       |
       |

This is the European/handwritten "1" style with an angled serif.

Supports both 28x28 (default) and 64x64 image sizes.

Output: data/custom_one/ directory with train/test NPZ files (80/20 split)
"""

import argparse
import numpy as np
import sys
import cv2
from pathlib import Path
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from DataManagement.DataCommon import (DATA_DIR)

OUTPUT_DIR = DATA_DIR / "custom_one"
# Number of variations to generate
NUM_VARIATIONS = 1250

# Train/test split
TRAIN_RATIO = 0.8


def draw_custom_one(size=28, stroke_width=2, diag_right=4, diag_up=3, vert_down=20, 
                    center_x=14, top_y=8):
    """
    Draw a custom "1" with diagonal flag at top.
    
    Shape:
          /|
         / |
      |
      |
      |
    
    Args:
        size: Image size (28 or 64)
        stroke_width: Width of strokes (scaled for size)
        diag_right: Pixels diagonal extends LEFT from vertical (positive = left)
        diag_up: Pixels diagonal extends UP (positive = up, Y decreases)
        vert_down: Pixels vertical extends DOWN (positive = down, Y increases)
        center_x: X position where vertical is centered
        top_y: Y position where vertical starts (top of vertical stroke)
    
    Returns:
        size x size numpy array with white digit on black background (grayscale)
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


def generate_variations(output_size=28):
    """Generate 5000 variations of the custom "1" with train/test split."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_images = []
    all_labels = []
    
    print(f"Generating {NUM_VARIATIONS} variations of custom '1'...")
    print(f"Output size: {output_size}x{output_size}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    for i in range(NUM_VARIATIONS):
        # Randomize parameters (scaled based on output_size)
        if output_size == 28:
            stroke_width = random.choice([1, 2, 2, 2, 3])
            diag_right = random.randint(3, 4)
            diag_up = random.randint(2, 5)
            vert_down = random.randint(18, 24)
            center_x = random.randint(12, 16)
            top_y = random.randint(6, 10)
        else:  # 64x64
            stroke_width = random.choice([2, 3, 4, 4, 5, 6])
            diag_right = random.randint(7, 10)
            diag_up = random.randint(5, 12)
            vert_down = random.randint(40, 55)
            center_x = random.randint(28, 38)
            top_y = random.randint(14, 22)
        
        # Draw base digit
        img = draw_custom_one(
            size=output_size,
            stroke_width=stroke_width,
            diag_right=diag_right,
            diag_up=diag_up,
            vert_down=vert_down,
            center_x=center_x,
            top_y=top_y
        )
        
        # Note: No rotation/shear here - handled by training augmentation
        
        # Keep as uint8 [0-255] - loader will normalize to [0,1]
        all_images.append(img)
        all_labels.append(1)  # Label is always 1
        
        # Save individual PNG (first 50 as samples)
        if i < 50:
            cv2.imwrite(str(OUTPUT_DIR / f"custom_one_{i:04d}.png"), img)
    
    # Convert to numpy arrays - keep as uint8 [0-255] and 3D (N, H, W)
    # load_dataset_from_npz will normalize and add channel dimension
    x_data = np.array(all_images, dtype=np.uint8)  # Shape: (N, output_size, output_size)
    y_softmax = np.array(all_labels, dtype=np.int32)  # Shape: (N,) - integer labels
        
    # Train/test split (80/20)
    print(f"\nSplitting into train/test ({int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)})...")
    
    # Shuffle indices
    np.random.seed(42)  # For reproducibility
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * TRAIN_RATIO)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Create train/test sets
    x_train = x_data[train_indices]
    y_train_softmax = y_softmax[train_indices]
    
    x_test = x_data[test_indices]
    y_test_softmax = y_softmax[test_indices]
    
    # Save train/test NPZ files (include size in filename)
    train_softmax_path = OUTPUT_DIR / f"custom_one_train_{output_size}x{output_size}.npz"
    test_softmax_path = OUTPUT_DIR / f"custom_one_test_{output_size}x{output_size}.npz"
    
    np.savez(train_softmax_path, x=x_train, y=y_train_softmax)
    np.savez(test_softmax_path, x=x_test, y=y_test_softmax)
    
    print(f"\n{'='*60}")
    print(f"Generated {len(all_images)} images at {output_size}x{output_size}")
    print(f"{'='*60}")
    print(f"Train set: {len(x_train):,} samples")
    print(f"Test set: {len(x_test):,} samples")
    print(f"\nSaved files:")
    print(f"  {train_softmax_path}")
    print(f"    x: {x_train.shape} uint8 [0-255], y: {y_train_softmax.shape} (integer labels)")
    print(f"  {test_softmax_path}")
    print(f"    x: {x_test.shape} uint8 [0-255], y: {y_test_softmax.shape} (integer labels)")
    print(f"\nSample PNGs: {OUTPUT_DIR}/custom_one_*.png (first 50)")
    print(f"{'='*60}")
    
    return (x_train, y_train_softmax), (x_test, y_test_softmax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate custom '1' digit variations")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Output image size (28 or 64, default: 28)")
    args = parser.parse_args()
    
    generate_variations(output_size=args.size)
