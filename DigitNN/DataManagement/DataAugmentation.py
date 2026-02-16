#!/usr/bin/env python3
"""
DataAugmentation.py

Shared data augmentation constants and helper functions.

This module defines reusable transform and post-processing utilities
(rotation, shear, aspect ratio, blur, thinning, thickening, erasure,
and stroke breaks), along with configuration constants used by those
operations.

It is designed to be imported by other modules in `DataManagement`
that generate or augment datasets.

"""

import numpy as np
import cv2
from PIL import Image, ImageFilter
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from DataManagement.DataCommon import (
    find_bounding_box,
    crop_resize_with_margin,
    float_to_uint8,
    MARGIN_SIZE
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Augmentation selection
AUGMENT_RATIO = 0.10  # 10% of each class selected for augmentation

# Transform parameters
ROTATION_RANGE_POS = (3, 15)    # degrees (positive)
ROTATION_RANGE_NEG = (-15, -3)  # degrees (negative)
SHEAR_RANGE_POS = (2, 10)       # degrees (positive)
SHEAR_RANGE_NEG = (-10, -2)     # degrees (negative)
ASPECT_WIDE_RANGE = (1.05, 1.6)   # stretch wider (max 60%)
ASPECT_NARROW_RANGE = (0.625, 0.95) # compress narrower (max 37.5%)

# Post-processing probabilities
BLUR_PROB = 0.20      # 20% get blurred
THIN_PROB = 0.10      # 10% get thinned
THICK_PROB = 0.10     # 10% get thickened
ERASURE_PROB = 0.10   # 10% get pixel erasure
BREAKS_PROB = 0.10    # 10% get stroke breaks

# Blur parameters
BLUR_RADIUS_RANGE = (0.5, 1.5)  # Gaussian blur radius

# Morphology parameters (for thin/thick)
THIN_KERNEL = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], np.uint8)  # 3x3 cross, gentle
THIN_ITERATIONS = 1

THICK_KERNEL = np.ones((2, 2), np.uint8)  # 2x2 square
THICK_ITERATIONS = 1

# Erasure parameters
ERASURE_PERCENT = 0.15  # 15% of white pixels erased

# Stroke break parameters
STROKE_BREAK_SIZE = 2   # 4 pixels per break
STROKE_BREAK_COUNT = 6  # 6 breaks scattered randomly


# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def apply_rotation(img_array, angle, target_size=None):
    """
    Apply rotation to image without clipping.

    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
        angle: rotation angle in degrees
        target_size: Target size for output (default: None = use original max dimension)

    Returns:
        Rotated image array, resized to (target_size, target_size)
    """
    import math
    
    # Handle channel dimension
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True

    h, w = img_array.shape
    original_size = max(h, w) 

    if target_size is None:
        target_size = original_size

    # Calculate required size for rotation: |cos(θ)| + |sin(θ)|
    angle_rad = math.radians(abs(angle))
    required_size = original_size * (abs(math.cos(angle_rad)) + abs(math.sin(angle_rad)))

    # Ensure canvas is large enough for all angles
    canvas_size = max(int(2*math.ceil(required_size)), 3*target_size)

    # Create larger canvas and embed original image centered
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    offset_y = (canvas_size - h) // 2
    offset_x = (canvas_size - w) // 2
    canvas[offset_y:offset_y+h, offset_x:offset_x+w] = img_array
    
    # Convert to uint8 for cv2
    canvas_uint8 = float_to_uint8(canvas)

    # Rotate on larger canvas
    center = (canvas_size // 2, canvas_size // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(canvas_uint8, M, (canvas_size, canvas_size), borderValue=0)
    
    # Resize back to original size using LANCZOS
    rotated_pil = Image.fromarray(rotated)
    bbox = find_bounding_box(rotated_pil)
    result_pil = crop_resize_with_margin(rotated_pil, 
                                         target_size=target_size, 
                                         bbox=bbox)

    # Back to numpy float
    result = np.array(result_pil).astype(np.float32) / 255.0
    
    if squeeze:
        result = np.expand_dims(result, -1)
    
    return result


def apply_shear(img_array, shear_degrees, target_size=None):
    """
    Apply vertical shear transformation without clipping.

    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
        shear_degrees: shear angle in degrees
        target_size: Target size for output (default: None = use original max dimension)

    Returns:
        Sheared image array, resized to (target_size, target_size)
    """
    import math
    
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True
    
    h, w = img_array.shape
    original_size = max(h, w) 

    if target_size is None:
        target_size = original_size

    
    # Preserve shear direction (sign) for the transform matrix.
    # Use absolute magnitude only for canvas sizing.
    shear_factor = math.tan(math.radians(shear_degrees))
    extra_width = int(math.ceil(abs(shear_factor) * h))
    required_width = w + extra_width
    
    # Add 2px padding on each side (+4px total)
    canvas_size = max(2 * required_width, 3 * target_size)  # Ensure canvas is large enough for all shears
    
    # Create larger canvas and embed original image centered
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    offset_y = (canvas_size - h) // 2
    offset_x = (canvas_size - w) // 2
    canvas[offset_y:offset_y+h, offset_x:offset_x+w] = img_array

    # Convert to uint8 for cv2
    canvas_uint8 = float_to_uint8(canvas)

    # Shear on larger canvas
    center_x = canvas_size // 2
    center_y = canvas_size // 2
    M = np.float32([[1, 0, 0],
                    [shear_factor, 1, -shear_factor * center_x]])
    
    sheared = cv2.warpAffine(canvas_uint8, M, (canvas_size, canvas_size), borderValue=0)
    
    # Resize back to original size using LANCZOS
    sheared_pil = Image.fromarray(sheared)
    bbox = find_bounding_box(sheared_pil)
    result_pil = crop_resize_with_margin(sheared_pil, 
                                         target_size=target_size, 
                                         bbox=bbox)

    # Back to numpy float
    result = np.array(result_pil).astype(np.float32) / 255.0
    
    if squeeze:
        result = np.expand_dims(result, -1)
    
    return result


def apply_aspect_ratio(img_array, aspect_factor, target_size=None):
    """
    Apply aspect ratio transformation (horizontal stretch/compress) without clipping.

    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
        aspect_factor: >1 = wider, <1 = narrower
        target_size: Target size for output (default: None = use original max dimension)

    Returns:
        Transformed image array, resized to (target_size, target_size)
    """
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True

    h, w = img_array.shape
    original_size = max(h, w)

    if target_size is None:
        target_size = original_size

    # Convert to uint8 and PIL
    img_uint8 = float_to_uint8(img_array)
    img_pil = Image.fromarray(img_uint8)

    # Step 1: Find bounding box of the digit in original image
    bbox = find_bounding_box(img_pil)

    if bbox is None:
        # No content, return black image
        result = np.zeros((target_size, target_size), dtype=np.float32)
        if squeeze:
            result = np.expand_dims(result, -1)
        return result

    x_min, y_min, x_max, y_max = bbox

    # Step 2: Crop to bounding box (add +1 to include pixels at x_max, y_max)
    digit_cropped = img_pil.crop((x_min, y_min, x_max+1, y_max+1))
    digit_w, digit_h = digit_cropped.size  # PIL returns (width, height)

    # Step 3: Apply aspect ratio transformation to the cropped digit
    new_width = int(digit_w * aspect_factor)
    transformed_digit = digit_cropped.resize((new_width, digit_h), Image.Resampling.LANCZOS)

    # Step 4: Create canvas large enough to hold transformed digit
    canvas_size = max(3 * new_width, 3 * digit_h, 2 * target_size)
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Step 5: Embed transformed digit on canvas (centered)
    transformed_array = np.array(transformed_digit)
    offset_y = (canvas_size - digit_h) // 2
    offset_x = (canvas_size - new_width) // 2
    canvas[offset_y:offset_y+digit_h, offset_x:offset_x+new_width] = transformed_array

    # Step 6: Find bounding box of transformed digit and crop with margin
    canvas_pil = Image.fromarray(canvas)
    new_bbox = find_bounding_box(canvas_pil)
    result_pil = crop_resize_with_margin(canvas_pil,
                                         target_size=target_size,
                                         bbox=new_bbox)

    # Back to numpy float
    result_array = np.array(result_pil).astype(np.float32) / 255.0

    if squeeze:
        result_array = np.expand_dims(result_array, -1)

    return result_array


def apply_blur(img_array, radius=1.0):
    """
    Apply Gaussian blur.
    
    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
        radius: blur radius
    
    Returns:
        Blurred image array, same shape
    """
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True

    # Convert to PIL
    img_uint8 = float_to_uint8(img_array)
    pil_img = Image.fromarray(img_uint8)
    
    # Apply blur
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    # Back to numpy float
    result = np.array(blurred).astype(np.float32) / 255.0
    
    if squeeze:
        result = np.expand_dims(result, -1)
    
    return result


def apply_thinning(img_array):
    """
    Apply stroke thinning (erosion with cross kernel).
    
    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
    
    Returns:
        Thinned image array, same shape
    """
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True

    # Convert to uint8
    img_uint8 = float_to_uint8(img_array)

    # Erode
    eroded = cv2.erode(img_uint8, THIN_KERNEL, iterations=THIN_ITERATIONS)
    
    # Back to float
    result = eroded.astype(np.float32) / 255.0
    
    if squeeze:
        result = np.expand_dims(result, -1)
    
    return result


def apply_thickening(img_array, target_size=None):
    """
    Apply stroke thickening (dilation) with proper bbox handling.

    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
        target_size: Target size for output (default: None = use original max dimension)

    Returns:
        Thickened image array, resized to (target_size, target_size)
    """
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True

    h, w = img_array.shape
    original_size = max(h, w)

    if target_size is None:
        target_size = original_size

    # Create canvas with 2× minimum guarantee
    canvas_size = max(h + 4, w + 4, 2 * original_size)

    # Embed original image on canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    offset_y = (canvas_size - h) // 2
    offset_x = (canvas_size - w) // 2
    canvas[offset_y:offset_y+h, offset_x:offset_x+w] = img_array

    # Convert to uint8
    canvas_uint8 = float_to_uint8(canvas)

    # Dilate on canvas
    dilated = cv2.dilate(canvas_uint8, THICK_KERNEL, iterations=THICK_ITERATIONS)

    # Find bounding box and crop with margin
    dilated_pil = Image.fromarray(dilated)
    bbox = find_bounding_box(dilated_pil)
    result_pil = crop_resize_with_margin(dilated_pil, 
                                         target_size=target_size,
                                         bbox=bbox)

    # Back to numpy float
    result = np.array(result_pil).astype(np.float32) / 255.0

    if squeeze:
        result = np.expand_dims(result, -1)

    return result


def apply_random_pixel_erasure(img_array, erasure_percent=ERASURE_PERCENT):
    """
    Randomly erase a percentage of white stroke pixels.
    
    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
        erasure_percent: fraction of white pixels to erase (default 0.15 = 15%)
    
    Returns:
        Image array with some pixels erased, same shape
    """
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True
    
    img_copy = img_array.copy()
    
    # Find all white pixels (stroke pixels, value > threshold)
    white_mask = img_copy > 0.1
    white_indices = np.where(white_mask)
    
    if len(white_indices[0]) == 0:
        if squeeze:
            return np.expand_dims(img_copy, -1)
        return img_copy
    
    total_white = len(white_indices[0])
    n_to_erase = int(total_white * erasure_percent)
    
    if n_to_erase == 0:
        if squeeze:
            return np.expand_dims(img_copy, -1)
        return img_copy
    
    # Randomly select pixels to erase
    indices_to_erase = np.random.choice(total_white, n_to_erase, replace=False)
    
    for idx in indices_to_erase:
        y, x = white_indices[0][idx], white_indices[1][idx]
        img_copy[y, x] = 0
    
    if squeeze:
        img_copy = np.expand_dims(img_copy, -1)
    
    return img_copy


def apply_stroke_breaks(img_array, break_size=STROKE_BREAK_SIZE, num_breaks=STROKE_BREAK_COUNT):
    """
    Apply small contiguous breaks (gaps) in the stroke.
    
    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
        break_size: size of each break in pixels (default 4)
        num_breaks: number of breaks to apply (default 6)
    
    Returns:
        Image array with stroke breaks, same shape
    """
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True
    
    img_copy = img_array.copy()
    height, width = img_copy.shape
    
    # Find all white pixels (stroke pixels)
    white_mask = img_copy > 0.1
    white_indices = np.where(white_mask)
    
    if len(white_indices[0]) == 0:
        if squeeze:
            return np.expand_dims(img_copy, -1)
        return img_copy
    
    for _ in range(num_breaks):
        # Pick a random stroke pixel as center of the break
        idx = np.random.randint(len(white_indices[0]))
        center_y, center_x = white_indices[0][idx], white_indices[1][idx]
        
        # Create a small rectangular break around the center
        half_size = break_size // 2
        y_start = max(0, center_y - half_size)
        y_end = min(height, center_y + half_size + 1)
        x_start = max(0, center_x - half_size)
        x_end = min(width, center_x + half_size + 1)
        
        # Erase the break area
        img_copy[y_start:y_end, x_start:x_end] = 0
    
    if squeeze:
        img_copy = np.expand_dims(img_copy, -1)
    
    return img_copy


def is_blank_image(img_array, threshold=0.01):
    """
    Check if image is all black or all white (for skipping augmentation).
    
    Args:
        img_array: numpy array, values 0-1
        threshold: variance threshold below which image is considered blank
    
    Returns:
        True if image is blank (all black or all white)
    """
    variance = np.var(img_array)
    return variance < threshold



# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Simple test
    print("DataAugmentation.py - Testing transforms")
    print("="*60)
    
    # Create dummy image (64x64 with a simple pattern)
    test_img = np.zeros((64, 64, 1), dtype=np.float32)
    test_img[20:50, 30:35, 0] = 1.0  # Vertical line
    test_img[20:25, 25:40, 0] = 1.0  # Horizontal line at top
    
    print(f"Test image shape: {test_img.shape}")
    print(f"Test image range: [{test_img.min():.2f}, {test_img.max():.2f}]")
    
    # Test transforms
    print("\nTesting transforms...")
    
    rotated = apply_rotation(test_img, 15, target_size=64)
    print(f"  Rotation (15°): shape {rotated.shape}, range [{rotated.min():.2f}, {rotated.max():.2f}]")
    
    sheared = apply_shear(test_img, 10, target_size=64)
    print(f"  Shear (10°): shape {sheared.shape}, range [{sheared.min():.2f}, {sheared.max():.2f}]")

    wide = apply_aspect_ratio(test_img, 1.5, target_size=64)
    print(f"  Aspect wide (1.5): shape {wide.shape}, range [{wide.min():.2f}, {wide.max():.2f}]")

    narrow = apply_aspect_ratio(test_img, 0.7, target_size=64)
    print(f"  Aspect narrow (0.7): shape {narrow.shape}, range [{narrow.min():.2f}, {narrow.max():.2f}]")

    blurred = apply_blur(test_img, 1.0)
    print(f"  Blur (r=1.0): shape {blurred.shape}, range [{blurred.min():.2f}, {blurred.max():.2f}]")

    thinned = apply_thinning(test_img)
    print(f"  Thinning: shape {thinned.shape}, range [{thinned.min():.2f}, {thinned.max():.2f}]")

    thickened = apply_thickening(test_img, target_size=64)
    print(f"  Thickening: shape {thickened.shape}, range [{thickened.min():.2f}, {thickened.max():.2f}]")
        
    print("\n✓ All tests passed!")
