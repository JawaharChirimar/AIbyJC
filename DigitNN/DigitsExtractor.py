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
from DigitClassifierNN import load_or_create_digit_classifier, classify_digit


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


def detect_background_color_contours(image):
    """
    Determine background color using contour-based detection.
    
    Finds foreground objects (digits) using contours, then samples pixels
    outside those contours to determine background color.
    
    Args:
        image: Input image (BGR format from cv2)
    
    Returns:
        bool: True if background is white (>127), False if black (<=127)
    """
    # Convert to greyscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply thresholding to find foreground objects (digits)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours (foreground objects)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask: foreground = 255, background = 0
    h, w = gray.shape
    foreground_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(foreground_mask, contours, -1, 255, -1)  # Fill contours
    
    # Background mask = inverse of foreground mask
    background_mask = 255 - foreground_mask
    
    # Sample background pixels from original grayscale image
    background_pixels = gray[background_mask > 0]
    
    if len(background_pixels) == 0:
        # No background pixels found, fall back to whole image mean
        mean_val = np.mean(gray)
        return mean_val > 127
    
    # Calculate mean of background pixels
    background_mean = np.mean(background_pixels)
    return background_mean


def detect_digits_with_contours(image, min_area=50, max_area=None, aspect_ratio_range=(0.2, 3.0)):
    """
    Detect digit regions using contour detection.
    
    This method uses OpenCV contour detection to find potential digit regions
    in the input image.
    
    Args:
        image: Input image (BGR format from cv2)
        min_area: Minimum contour area to consider (filters out noise)
        max_area: Maximum contour area (None = no limit)
        aspect_ratio_range: (min, max) aspect ratio for bounding boxes
    
    Returns:
        List of detection boxes (x1, y1, x2, y2)
    """
    # Convert to greyscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive thresholding or Otsu's thresholding
    # Try Otsu's first, fall back to adaptive if needed
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Alternative: use adaptive thresholding for varying lighting
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    h, w = gray.shape
    
    # Set default max_area if not provided (e.g., 10% of image area)
    if max_area is None:
        max_area = (w * h) * 0.1
    
    for contour in contours:
        # Get bounding rectangle
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = box_w * box_h
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Calculate aspect ratio
        if box_h > 0:
            aspect_ratio = box_w / box_h
        else:
            continue
        
        # Filter by aspect ratio (digits are usually not too wide or too tall)
        if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
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
background_mean=0):
    """
    Extract region from image, resize to target size, and convert to greyscale.
    
    Args:
        image: Input image (BGR format from cv2)
        box: Bounding box (x1, y1, x2, y2)
        target_size: Target size (width, height)
        background_mean: mean value of background (determined from full image)
    
    Returns:
        Processed 56x56 greyscale image
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

    target_size = 38

    digit1 = cv2.resize(digit0, (target_size, target_size), interpolation=cv2.INTER_AREA)   
    
    # Light noise reduction - very gentle blur to smooth without losing detail
    digit2 = cv2.bilateralFilter(digit1, 5, 50, 50)
    

    digit3 = cv2.adaptiveThreshold(digit2, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)

    # Very light noise removal - only remove tiny isolated pixels (only for binary)
    kernel = np.ones((1, 1), np.uint8)  # Minimal kernel
    digit4 = cv2.morphologyEx(digit3, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # heigth and width of original bounding box
    h, w = image[y1:y2, x1:x2].shape[:2]
    
    # Calculate scaling factor to fit the larger dimension to target_size
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize maintaining aspect ratio
    digit5 = cv2.resize(digit4, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Add padding to make it square, centering the digit in the target_size
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    pad_h_remainder = (target_size - new_h) % 2  # Handle odd padding
    pad_w_remainder = (target_size - new_w) % 2
    
    digit6 = cv2.copyMakeBorder(
        digit5,
        top=pad_h,
        bottom=pad_h + pad_h_remainder,
        left=pad_w,
        right=pad_w + pad_w_remainder,
        borderType=cv2.BORDER_CONSTANT,
        value=background_mean  # Color matches background mean
    )
    
    digit7 = np.where(digit6 > (background_mean - 1), 0, 255).astype(np.uint8)
    
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

    digit9 = cv2.resize(digit8, (28, 28), interpolation=cv2.INTER_CUBIC)

    #digit10 = cv2.adaptiveThreshold(digit9, 255, 
    #cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)

    # Very light noise removal - only remove tiny isolated pixels (only for binary)
    #kernel = np.ones((1, 1), np.uint8)  # Minimal kernel
    #digit11 = cv2.morphologyEx(digit10, cv2.MORPH_OPEN, kernel, iterations=1)
    digit11 = digit9
    
    return digit11


def process_image(input_path, output_dir=None, 
classifier_model_path=None, classify_digits=False):
    """
    Process input image with contour detection, extract digit regions, and save them.
    
    Args:
        input_path: Path to input JPEG file
        output_dir: Output directory (created with timestamp if None)
        classifier_model_path: Path to digit classifier model (optional)
        classify_digits: Whether to classify digits using CNN
    """
    # Load image
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    image2 = cv2.imread(input_path)
    image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    if image is None:
        print(f"Error: Could not read image: {input_path}")
        sys.exit(1)
    
    image_height, image_width = image.shape[:2]
    
    # Detect background color using contour-based detection
    background_mean = detect_background_color_contours(image)
    
    # Detect digits using contour-based detection
    detections = detect_digits_with_contours(image, min_area=30, 
    aspect_ratio_range=(0.15, 4.0))
    
    if not detections:
        print("Error: No digit regions found using contour detection.")
        print("Please try:")
        print("  1. Adjusting image preprocessing (ensure good contrast)")
        print("  2. Checking if the image contains visible handwritten digits")
        return
    else:
        print(f"Found {len(detections)} potential digit regions using contour detection")
    
    # Sort detections in reading order
    sorted_detections = sort_detections_by_reading_order(detections, image_height)
    
    # Create output directory
    if output_dir is None:
        output_dir = create_output_directory()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load digit classifier if classification is requested
    classifier_model = None
    if classify_digits:
        try:
            classifier_model = load_or_create_digit_classifier(train_model=False, classifier_model_path=classifier_model_path)
        except Exception as e:
            print(f"Warning: Could not load/create digit classifier: {e}")
            print("Saving regions without classification...")
            classify_digits = False
    
    # Process and save each region
    for line_num, digit_num, box in sorted_detections:
        # Extract and process region
        processed_region = extract_and_process_region(image, box, 
        background_mean=background_mean)
        
        # Classify digit if requested
        predicted_digit = None
        confidence = None
        if classify_digits and classifier_model:
            try:
                predicted_digit, confidence = classify_digit(classifier_model, processed_region)
            except Exception as e:
                print(f"Warning: Classification failed for region {line_num}_{digit_num}: {e}")
        
        # Create filename: file_L_D.jpg (or file_L_D_classified_X.jpg if classifying)
        if predicted_digit is not None:
            filename = f"file_{line_num}_{digit_num}_classified_{predicted_digit}_conf_{confidence:.2f}.jpg"
        else:
            filename = f"file_{line_num}_{digit_num}.jpg"
        
        output_path = output_dir / filename
        
        # Save as JPEG
        cv2.imwrite(str(output_path), processed_region, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if predicted_digit is not None:
            print(f"Saved: {filename} (predicted: {predicted_digit}, confidence: {confidence:.2%})")
        else:
            print(f"Saved: {filename}")
    
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

