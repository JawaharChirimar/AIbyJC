#!/usr/bin/env python3
"""
DigitsExtractor.py

Processes a JPEG image with handwritten digits using contour detection.
Extracts each detected digit region, normalizes to 28x28 greyscale, and saves
as individual JPEG files with naming pattern: file_L_D.jpg
where L = line number (0-indexed), D = digit number (1-indexed).

Output directory: ~/Development/AIbyJC/DigitNN/data/DERun/run_YYYY_MM_DD_HH_MM_SS/
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from DigitClassifierALL import load_or_create_digit_classifier, classify_digit


def create_output_directory():
    """Create output directory with timestamp in ~/Development/AIbyJC/DigitNN/data/DERun/"""
    home_dir = Path.home()
    output_base = home_dir / "Development" / "AIbyJC" / "DigitNN" /"data" / "DERun"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = output_base / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def sort_detections_by_reading_order(detections, image_height, line_threshold=0.1):
    """
    Sort detections in reading order: top to bottom, left to right.
    
    Groups detections into lines based on y-coordinate similarity,
    then sorts each line by x-coordinate.
    
    Args:
        detections: List of detection boxes (x1, y1, x2, y2)
        image_height: Height of the image (for relative thresholding)
        line_threshold: Relative threshold for grouping into lines (fraction of image height)
    
    Returns:
        List of (line_number, digit_index, box) tuples
    """
    if not detections:
        return []
    
    # Calculate absolute threshold based on image height
    threshold = image_height * line_threshold
    
    # Group boxes by line (similar y-coordinates)
    lines = []
    for box in detections:
        x1, y1, x2, y2 = box
        center_y = (y1 + y2) / 2
        
        # Find existing line or create new one
        assigned = False
        for line_boxes in lines:
            # Check if this box belongs to this line (based on y-coordinate)
            if line_boxes:
                line_center_y = np.mean([(b[1] + b[3]) / 2 for b in line_boxes])
                if abs(center_y - line_center_y) < threshold:
                    line_boxes.append(box)
                    assigned = True
                    break
        
        if not assigned:
            lines.append([box])
    
    # Sort lines by topmost y-coordinate
    lines.sort(key=lambda line_boxes: min([b[1] for b in line_boxes]))
    
    # Sort boxes within each line by x-coordinate
    for line_boxes in lines:
        line_boxes.sort(key=lambda b: b[0])
    
    # Create result list with (line_number, digit_index, box)
    # line_number is 0-indexed, digit_index is 1-indexed
    result = []
    for line_idx, line_boxes in enumerate(lines):
        for digit_idx, box in enumerate(line_boxes, start=1):
            result.append((line_idx, digit_idx, box))
    
    return result


def get_foreground_mask(image):
    """
    Create binary mask with foreground=255, background=0.
    
    Constraint: foreground must be < 50% of image area.
    
    Raises:
        ValueError: If foreground/background cannot be determined.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    white_ratio = np.mean(binary) / 255
    
    # Clear cases
    if white_ratio > 0.5:
        binary = cv2.bitwise_not(binary)
        white_ratio = 1 - white_ratio
    
    # Ambiguous case - too close to 50%
    if white_ratio > 0.45:
        raise ValueError(
            f"Cannot determine foreground/background. "
            f"Foreground ratio: {white_ratio:.1%}. "
            f"Ensure text covers < 50% of image."
        )
    
    return binary


def detect_background_color_contours(image):
    """
    Determine background and foreground colors using contour-based detection.
    
    Finds foreground objects (digits) using get_foreground_mask, then samples pixels
    to determine background and foreground colors.
    
    Args:
        image: Input image (BGR format from cv2)
    
    Returns:
        tuple: (background_mean, foreground_mean) - mean pixel values
    """
    # Convert to greyscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Get foreground mask (foreground=255, background=0)
    foreground_mask = get_foreground_mask(image)
    
    # Background mask = inverse of foreground mask
    background_mask = 255 - foreground_mask
    
    # Sample background pixels from original grayscale image
    background_pixels = gray[background_mask > 0]
    
    if len(background_pixels) == 0:
        # This should never happen if get_foreground_mask succeeded
        # (foreground < 50% means background > 50%)
        raise ValueError(
            "No background pixels found. This indicates an error in foreground/background detection. "
            "Ensure text covers < 50% of image area."
        )
    
    # Sample foreground pixels from original grayscale image
    foreground_pixels = gray[foreground_mask > 0]
    
    if len(foreground_pixels) == 0:
        # No foreground pixels - assume same as background
        foreground_mean = np.mean(background_pixels)
    else:
        foreground_mean = np.mean(foreground_pixels)
    
    # Calculate mean of background pixels
    background_mean = np.mean(background_pixels)
    return background_mean, foreground_mean


def detect_digits_with_contours(image, min_area=50):
    """
    Detect digit regions using contour detection.
    
    This method uses OpenCV contour detection to find potential digit regions
    in the input image.
    
    Args:
        image: Input image (BGR format from cv2)
        min_area: Minimum contour area to consider (filters out noise)
    
    Returns:
        List of detection boxes (x1, y1, x2, y2)
    
    Raises:
        ValueError: If foreground/background cannot be determined (foreground > 45%).
    """
    # Convert to greyscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Get foreground mask (foreground=255, background=0)
    binary = get_foreground_mask(image)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    h, w = gray.shape
    
    for contour in contours:
        # Get bounding rectangle
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = box_w * box_h
        
        # Filter by minimum area (removes noise)
        if area < min_area:
            continue
        
        # Add padding around the bounding box
        padding = 2
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + box_w + padding)
        y2 = min(h, y + box_h + padding)
        
        detections.append((float(x1), float(y1), float(x2), float(y2)))
    
    return detections


def extract_and_process_region(image, box, target_size=(28, 28), 
background_mean=0, foreground_mean=255):
    """
    Extract region from image, scale to 28x28 first, then apply transformations.
    
    Args:
        image: Input image (BGR format from cv2)
        box: Bounding box (x1, y1, x2, y2)
        target_size: Target size (width, height)
        background_mean: mean value of background (determined from full image)
        foreground_mean: mean value of foreground/digits (determined from full image)
    
    Returns:
        Processed 28x28 binary image
    """
    import sys
    print("DEBUG: extract_and_process_region called", file=sys.stderr, flush=True)
    x1, y1, x2, y2 = map(int, box)
    
    # Determine if digits are darker or lighter than background
    digits_are_darker = foreground_mean < background_mean
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Extract region
    region = image[y1:y2, x1:x2]
    
    # Convert to grayscale if needed
    if len(region.shape) == 3:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # === STEP 1: Scale to 28x28 maintaining aspect ratio ===
    region_h, region_w = region.shape[:2]
    
    # Skip very small regions
    if region_h < 2 or region_w < 2:
        # Return a blank 28x28 image
        return np.zeros(target_size, dtype=np.uint8)
    
    # Leave margin for padding (digit centered with some border)
    margin = 4
    max_dim = max(region_h, region_w)
    scale = (target_size[0] - margin) / max_dim
    new_w = max(1, int(region_w * scale))  # Ensure at least 1 pixel
    new_h = max(1, int(region_h * scale))  # Ensure at least 1 pixel
    
    # Resize maintaining aspect ratio
    scaled = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Add padding to center and make it target_size
    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2
    pad_h_remainder = (target_size[0] - new_h) % 2
    pad_w_remainder = (target_size[1] - new_w) % 2
    
    # Pad with background color
    pad_value = int(background_mean)
    digit_28x28 = cv2.copyMakeBorder(
        scaled,
        top=pad_h,
        bottom=pad_h + pad_h_remainder,
        left=pad_w,
        right=pad_w + pad_w_remainder,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value
    )
    
    # === STEP 2: Apply transformations on 28x28 image ===
    
    # Light noise reduction
    filtered = cv2.bilateralFilter(digit_28x28, 5, 50, 50)
    
    # Dilate to preserve stroke thickness before thresholding
    dilate_kernel = np.ones((1, 1), np.uint8)
    if not digits_are_darker:
        # Digits are lighter than background: dilate to expand bright digits
        dilated = cv2.dilate(filtered, dilate_kernel, iterations=1)
    else:
        # Digits are darker than background: erode to expand dark digits
        dilated = cv2.erode(filtered, dilate_kernel, iterations=1)
    
    # Threshold - with offset to include more grays as white (digit pixels)
    # Higher offset = lower threshold = more grays become white
    threshold_offset = 1  # Adjust this value: 0-50, higher = more grays become white
    
    if not digits_are_darker:
        # Digits are lighter than background - use Otsu's threshold
        otsu_thresh, _ = cv2.threshold(dilated, 0, 255, cv2.THRESH_OTSU)
        adjusted_thresh = max(0, otsu_thresh - threshold_offset)
        _, binary = cv2.threshold(dilated, adjusted_thresh, 255, cv2.THRESH_BINARY)
    else:
        # Digits are darker than background - use adaptive threshold with inversion
        # C parameter: higher = more pixels become white after inversion
        adaptive_C = 5 + (threshold_offset // 4)  # Scale offset for adaptive
        binary = cv2.adaptiveThreshold(dilated, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, adaptive_C)
    
    # Morphological closing to reconnect disconnected parts
    close_kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    
    # Light noise removal
    open_kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=1)
    
    # Return grayscale instead of binary to preserve information for the model
    # The model was trained on grayscale images (0-255), not binary
    # Convert binary: white digits stay white (255), black background becomes dark grey
    final_binary = np.where(cleaned > 127, 255, 30).astype(np.uint8)  # Black (0) -> dark grey (30)
    
    # DEBUG: Check before blur
    import sys
    unique_before = np.unique(final_binary)
    print(f"DEBUG: Before blur - unique values: {len(unique_before)}, range: {unique_before.min()}-{unique_before.max()}, values: {unique_before[:10]}", file=sys.stderr, flush=True)
    
    # Apply Gaussian blur to create smooth grayscale transitions
    # Use (3, 3) kernel with sigma 0.5 to match Mac behavior
    final = cv2.GaussianBlur(final_binary, (3, 3), 0.6)
    
    # DEBUG: Check after blur
    unique_after = np.unique(final)
    print(f"DEBUG: After blur - unique values: {len(unique_after)}, range: {unique_after.min()}-{unique_after.max()}, sample values: {unique_after[:20]}", file=sys.stderr, flush=True)
    
    # DEBUG: Check if JPEG encoding affects it
    success, buffer = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if success:
        decoded = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        unique_after_jpeg = np.unique(decoded)
        print(f"DEBUG: After JPEG encode/decode - unique values: {len(unique_after_jpeg)}, range: {unique_after_jpeg.min()}-{unique_after_jpeg.max()}, sample values: {unique_after_jpeg[:20]}", file=sys.stderr, flush=True)
    
    return final


def extract_and_process_region_grayscale(image, box, target_size=(28, 28), 
background_mean=0, foreground_mean=255):
    """
    Extract region from image, scale to 28x28, and process while preserving grayscale throughout.
    This version does NOT threshold to binary, keeping grayscale values for the model.
    
    Args:
        image: Input image (BGR format from cv2)
        box: Bounding box (x1, y1, x2, y2)
        target_size: Target size (width, height)
        background_mean: mean value of background (determined from full image)
        foreground_mean: mean value of foreground/digits (determined from full image)
    
    Returns:
        Processed 28x28 grayscale image (0-255, not binary)
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Determine if digits are darker or lighter than background
    digits_are_darker = foreground_mean < background_mean
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Extract region
    region = image[y1:y2, x1:x2]
    
    # Convert to grayscale if needed
    if len(region.shape) == 3:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # === STEP 1: Scale to 28x28 maintaining aspect ratio ===
    region_h, region_w = region.shape[:2]
    
    # Skip very small regions
    if region_h < 2 or region_w < 2:
        # Return a blank 28x28 image
        return np.zeros(target_size, dtype=np.uint8)
    
    # Leave margin for padding (digit centered with some border)
    margin = 4
    max_dim = max(region_h, region_w)
    scale = (target_size[0] - margin) / max_dim
    new_w = max(1, int(region_w * scale))
    new_h = max(1, int(region_h * scale))
    
    # Resize maintaining aspect ratio
    scaled = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Add padding to center and make it target_size
    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2
    pad_h_remainder = (target_size[0] - new_h) % 2
    pad_w_remainder = (target_size[1] - new_w) % 2
    
    # Pad with background color
    pad_value = int(background_mean)
    digit_28x28 = cv2.copyMakeBorder(
        scaled,
        top=pad_h,
        bottom=pad_h + pad_h_remainder,
        left=pad_w,
        right=pad_w + pad_w_remainder,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value
    )
    
    # === STEP 2: Apply transformations on 28x28 grayscale image (NO THRESHOLDING) ===
    
    # First, ensure MNIST/EMNIST format: white digits on black background
    # If digits are darker than background, invert the image
    if digits_are_darker:
        # Invert: dark digits on light background -> light digits on dark background
        digit_28x28 = 255 - digit_28x28
    
    # Very light noise reduction - minimal to avoid thickening
    filtered = cv2.bilateralFilter(digit_28x28, 3, 30, 30)
    
    # Increase contrast to match EMNIST format (high contrast, uniform black background)
    # Use percentile-based stretching to preserve digit structure
    p20, p80 = np.percentile(filtered, (20, 80))
    if p80 > p20:
        # Map [p20, p80] to [0, 255] - gentler stretching to preserve shape
        contrast_stretched = np.clip((filtered.astype(np.float32) - p20) * 255.0 / (p80 - p20), 0, 255).astype(np.uint8)
    else:
        contrast_stretched = filtered
    
    # Identify background using a lower threshold to avoid cutting digit strokes
    # Use a percentile-based threshold instead of Otsu to be less aggressive
    bg_threshold = np.percentile(contrast_stretched, 25)  # Bottom 25% are background
    background_mask = contrast_stretched < bg_threshold
    
    # Light morphological closing on digit pixels only to reconnect cut strokes
    # Create a mask for digit pixels
    digit_mask = ~background_mask
    if np.any(digit_mask):
        # Apply closing only to digit region to reconnect broken strokes
        kernel = np.ones((2, 2), np.uint8)
        # Create temporary binary for closing
        temp_binary = np.zeros_like(contrast_stretched)
        temp_binary[digit_mask] = contrast_stretched[digit_mask]
        closed = cv2.morphologyEx(temp_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Blend closed result back into digit pixels only
        contrast_stretched[digit_mask] = np.maximum(contrast_stretched[digit_mask], closed[digit_mask] * 0.7).astype(np.uint8)
    
    # Force all background pixels to uniform black (0)
    # This ensures clean, uniform black background like EMNIST
    contrast_stretched[background_mask] = 0
    
    # Skip LUT and morphological operations - they thicken strokes
    # Just apply minimal smoothing to create smooth grayscale transitions
    # Very light blur to preserve exact digit shape
    final = cv2.GaussianBlur(contrast_stretched, (3, 3), 0.15)
    
    return final


def extract_and_process_region_prev(image, box, target_size=(28, 28), 
background_mean=0):
    """
    PREVIOUS VERSION: Extract region, transform, then resize multiple times.
    
    Args:
        image: Input image (BGR format from cv2)
        box: Bounding box (x1, y1, x2, y2)
        target_size: Target size (width, height)
        background_mean: mean value of background (determined from full image)
    
    Returns:
        Processed 28x28 binary image
    """
    x1, y1, x2, y2 = map(int, box)

    print("background_mean: ", background_mean)
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Extract region
    digit0 = image[y1:y2, x1:x2]
    # Convert to grayscale if needed
    if len(digit0.shape) == 3:
        digit0 = cv2.cvtColor(digit0, cv2.COLOR_BGR2GRAY)


    digit1 = cv2.resize(digit0, target_size, interpolation=cv2.INTER_AREA)   
    
    # Light noise reduction - very gentle blur to smooth without losing detail
    digit2 = cv2.bilateralFilter(digit1, 5, 50, 50)
    
    # Threshold based on background type
    # Dilate grayscale image BEFORE thresholding to preserve thick strokes
    # For dark background: dilate bright pixels (digits)
    # For light background: erode (which dilates dark pixels = digits)
    dilate_kernel = np.ones((3, 3), np.uint8)  # Larger kernel for thin strokes
    if background_mean < 127:
        # Dark background: dilate to expand white digits
        digit2_dilated = cv2.dilate(digit2, dilate_kernel, iterations=1)
    else:
        # Light background: erode to expand dark digits (before inversion)
        digit2_dilated = cv2.erode(digit2, dilate_kernel, iterations=1)

    if background_mean < 127:
        # Dark background with light digits (white-on-black)
        # Use simple Otsu's threshold - adaptive doesn't work well for this case
        _, digit3 = cv2.threshold(digit2_dilated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Light background with dark digits (black-on-white)
        # Use adaptive threshold, then invert to get white-on-black (MNIST format)
        digit3 = cv2.adaptiveThreshold(digit2_dilated, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    
    # Morphological closing to reconnect nearby disconnected parts
    close_kernel = np.ones((3, 3), np.uint8)
    digit3 = cv2.morphologyEx(digit3, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Very light noise removal - only remove tiny isolated pixels (only for binary)
    kernel = np.ones((1, 1), np.uint8)  # Minimal kernel
    digit4 = cv2.morphologyEx(digit3, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Height and width of original bounding box
    h, w = image[y1:y2, x1:x2].shape[:2]
    
    # Calculate scaling factor to fit the larger dimension to target_size
    # Leave some margin for padding (e.g., 4 pixels)
    margin = 4
    max_dim = max(h, w)
    scale = (target_size[0] - margin) / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize maintaining aspect ratio
    digit5 = cv2.resize(digit4, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Add padding to make it square, centering the digit in the target_size
    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2
    pad_h_remainder = (target_size[0] - new_h) % 2  # Handle odd padding
    pad_w_remainder = (target_size[1] - new_w) % 2
    
    # Add padding with black (0) since we want white digits on black background
    digit6 = cv2.copyMakeBorder(
        digit5,
        top=pad_h,
        bottom=pad_h + pad_h_remainder,
        left=pad_w,
        right=pad_w + pad_w_remainder,
        borderType=cv2.BORDER_CONSTANT,
        value=0  # Black padding (MNIST format: white digits on black background)
    )
    
    # Ensure binary: anything that's not white (255) becomes black (0)
    # digit5 is already binary (0 or 255) from adaptiveThreshold, but padding might have added grayscale
    digit7 = np.where(digit6 > 127, 255, 0).astype(np.uint8)
    
    # Add 4 pixels of solid black padding on all 4 sides to make 28x28
    # Final size: 52 + 2 + 2 = 56
    padding = 2
    digit8 = cv2.copyMakeBorder(
        digit7,
        top=padding,
        bottom=padding,
        left=padding,
        right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=0  # Black padding
    )
    
    return digit8



def process_image(input_path=None, output_dir=None, 
classifier_model_path=None, classify_digits=False, 
image_array=None, return_results=False):
    """
    Process input image with contour detection, extract digit regions, and save them or return results.
    
    Args:
        input_path: Path to input JPEG file (required if image_array is None)
        output_dir: Output directory (created with timestamp if None, required if return_results=False)
        classifier_model_path: Path to digit classifier model (required if classify_digits=True)
        classify_digits: Whether to classify digits using CNN (required if return_results=True)
        image_array: numpy array of image in BGR format (optional, if provided, input_path is ignored)
        return_results: If True, return list of dicts instead of saving files (default: False)
    
    Returns:
        If return_results=True: dict with 'error' key OR 'results' key containing list of dicts:
            - Each dict: {'image': base64_encoded_image, 'digit': int, 'confidence': float}
        If return_results=False: None (saves files to disk)
    """
    import base64
    
    # Load image - either from array or file path
    if image_array is not None:
        # Image already provided as array (BGR format)
        image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        # Load from file path (existing behavior)
        if input_path is None:
            if return_results:
                return {'error': 'Either input_path or image_array must be provided'}
            print("Error: Input file path is required")
            sys.exit(1)
        if not os.path.exists(input_path):
            if return_results:
                return {'error': f'Input file not found: {input_path}'}
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        
        image2 = cv2.imread(input_path)
        image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        if image is None:
            if return_results:
                return {'error': f'Could not read image: {input_path}'}
            print(f"Error: Could not read image: {input_path}")
            sys.exit(1)
    
    image_height, image_width = image.shape[:2]
    
    # Detect background and foreground colors using contour-based detection
    background_mean, foreground_mean = detect_background_color_contours(image)
    if not return_results:
        print(f"Background mean: {background_mean:.1f}, Foreground mean: {foreground_mean:.1f}")
    
    # Detect digits using contour-based detection
    detections = detect_digits_with_contours(image, min_area=30)
    
    if not detections:
        error_msg = "No digit regions found. Please ensure the image contains visible handwritten digits."
        if return_results:
            return {'error': error_msg}
        print("Error: No digit regions found using contour detection.")
        print("Please try:")
        print("  1. Adjusting image preprocessing (ensure good contrast)")
        print("  2. Checking if the image contains visible handwritten digits")
        return
    
    if not return_results:
        print(f"Found {len(detections)} potential digit regions using contour detection")
    
    # Sort detections in reading order
    sorted_detections = sort_detections_by_reading_order(detections, image_height)
    
    # Create output directory (only needed if not returning results)
    if not return_results:
        if output_dir is None:
            output_dir = create_output_directory()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Load digit classifier if classification is requested (required for return_results)
    classifier_model = None
    if classify_digits or return_results:
        if classifier_model_path is None:
            if return_results:
                return {'error': 'classifier_model_path is required when return_results=True'}
            print("Warning: classifier_model_path not provided, skipping classification")
            classify_digits = False
        else:
            try:
                classifier_model = load_or_create_digit_classifier(train_model=False, classifier_model_path=classifier_model_path)
            except Exception as e:
                if return_results:
                    return {'error': f'Could not load classifier model: {str(e)}'}
                print(f"Warning: Could not load/create digit classifier: {e}")
                print("Saving regions without classification...")
                classify_digits = False
    
    # Process each region
    results = []
    for line_num, digit_num, box in sorted_detections:
        # Extract and process region (using original pipeline with thresholding)
        processed_region = extract_and_process_region(image, box, 
        background_mean=background_mean, foreground_mean=foreground_mean)
        
        # Classify digit if requested
        predicted_digit = None
        confidence = None
        if (classify_digits or return_results) and classifier_model:
            try:
                predicted_digit, confidence = classify_digit(classifier_model, processed_region)
            except Exception as e:
                if return_results:
                    return {'error': f'Classification failed: {str(e)}'}
                print(f"Warning: Classification failed for region {line_num}_{digit_num}: {e}")
        
        if return_results:
            # Encode processed region (28x28 image that model saw) as base64
            success, buffer = cv2.imencode('.jpg', processed_region, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                return {'error': 'Failed to encode processed image'}
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Handle rejected digits (None or -1) - convert to -1 for API
            digit_value = -1 if (predicted_digit is None or predicted_digit == -1) else int(predicted_digit)
            
            results.append({
                'image': image_base64,
                'digit': digit_value,
                'confidence': float(confidence) if confidence is not None else 0.0
            })
        else:
            # Save to file (existing behavior)
            # Create filename: file_L_D.jpg (or file_L_D_classified_X.jpg if classifying)
            # Handle both None (from sigmoid models) and -1 (from softmax 11-class model) as rejected
            if predicted_digit is not None and predicted_digit != -1:
                filename = f"file_{line_num}_{digit_num}_classified_{predicted_digit}_conf_{confidence:.2f}.jpg"
            else:
                filename = f"file_{line_num}_{digit_num}.jpg"
            
            output_path = output_dir / filename
            
            # Save as JPEG
            cv2.imwrite(str(output_path), processed_region, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if predicted_digit is not None and predicted_digit != -1:
                print(f"Saved: {filename} (predicted: {predicted_digit}, confidence: {confidence:.2%})")
            else:
                print(f"Saved: {filename}")
    
    if return_results:
        return {'results': results}
    else:
        print(f"\nProcessing complete! Saved {len(sorted_detections)} digit regions to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract handwritten digits from JPEG image using contour detection"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to input JPEG file with handwritten digits"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (optional, creates timestamped directory if not provided)"
    )
    parser.add_argument(
        "-c", "--classify",
        action="store_true",
        help="Enable digit classification using CNN (requires TensorFlow, will train on MNIST if no model provided)"
    )
    parser.add_argument(
        "-m", "--classifier-model",
        type=str,
        default=None,
        help="Path to pre-trained digit classifier model (.keras file). Required when --classify is used."
    )
    
    args = parser.parse_args()
    
    # Validate that classifier-model is provided when classification is enabled
    if args.classify and args.classifier_model is None:
        parser.error("--classifier-model is required when --classify is used")
    
    process_image(args.input_image, args.output, args.classifier_model, args.classify)


if __name__ == "__main__":
    main()

