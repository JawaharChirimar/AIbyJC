#!/usr/bin/env python3
"""
FontDigitGenerator.py

Downloads fonts from Google Fonts API and generates augmented digit images (0-9)
for use as training data.

Font Selection:
- Top 75 handwriting fonts by popularity (non-italic) - most similar to handwritten digits
- Top 20 sans-serif fonts by popularity (non-italic) - clean digit shapes
- Top 10 display fonts by popularity (non-italic) - varied styles
- 1 weight per font (preferring regular)

Each font generates 1,090 images:
- 10 digits × 109 variations each
- 27 combinations (stroke × aspect) × 4 types (rotation, shear, erasure, breaks) + 1 original
- 4 variation types: rotation, shear, 15% pixel erasure, 6 stroke breaks (4px)

Usage:
    python FontDigitGenerator.py --output-dir ./data/font_digits
    
Output:
    - Individual PNG images
    - font_digits_softmax.npz: x (N, 64, 64, 1), y (N,) integer labels
    - font_digits_sigmoid.npz: x (N, 64, 64, 1), y (N, 10) one-hot labels
"""

import os
import argparse
import requests
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import tempfile
import cv2


# =============================================================================
# CONFIGURATION
# =============================================================================
# Google Fonts API Key - Get from: https://console.cloud.google.com/
# Enable "Google Fonts Developer API" and create an API key
# We get it either from the command line 
# or from the environment variable GOOGLE_FONTS_API_KEY

DIGITS = "0123456789"
# TARGET_SIZE will be set dynamically based on --size argument

# Font category limits
HANDWRITING_LIMIT = 75  # Top N handwriting fonts by popularity
SANS_SERIF_LIMIT = 20   # Top N sans-serif fonts by popularity
DISPLAY_LIMIT = 10      # Top N display fonts by popularity

# Font category configuration:
# - handwriting: Top N by popularity (most valuable for digit recognition)
# - sans-serif: Top N by popularity (clean shapes)
# - display: Top N by popularity (varied styles, more conservative)
FONT_CATEGORY_LIMITS = {
    "handwriting": HANDWRITING_LIMIT,
    "sans-serif": SANS_SERIF_LIMIT,
    "display": DISPLAY_LIMIT,
}

# Font weights: use 1 weight per font (non-italic only)
# Priority: regular (400) > bold (700) > thin (100-300)
PREFERRED_WEIGHTS = ["100", "200", "300", "regular", "400", "500", "600", "700", "800", "900"]
# Exclude italic variants
EXCLUDE_VARIANTS = ["italic", "100italic", "200italic", "300italic", "400italic", 
                    "500italic", "600italic", "700italic", "800italic", "900italic"]

# Output format:
# - Size: 64x64 pixels
# - Channels: 1 (grayscale)
# - Background: Black (0)
# - Digit: White (normalized to 0-1 range)
# - Dtype: float32
# - Shape: (N, 64, 64, 1)

# Augmentation parameters
ROTATION_RANGE_NEG = (-30, -3)   # degrees (negative rotation range)
ROTATION_RANGE_POS = (3, 30)      # degrees (positive rotation range)
SHEAR_RANGE_NEG = (-16, -2)      # degrees vertical shear (negative range)
SHEAR_RANGE_POS = (2, 16)         # degrees vertical shear (positive range)
NOISE_STD = 0.05                 # Gaussian noise std dev
STROKE_VARIATIONS = [-1, 0, 1]   # -1 = thinner (erode), 0 = normal, 1 = thicker (dilate)
ASPECT_RATIOS = [0.625, 0.75, 0.85, 0.90, 1.00, 1.125, 1.25, 1.50, 1.60]  # width stretch factors

# Combinations: stroke (3) × aspect (9) = 27 combinations
# Per combination: 1 rotation + 1 shear + 1 erasure + 1 stroke breaks = 4 variations
# Plus 1 original per digit = 109 total per digit
COMBINATIONS_COUNT = 27  # 3 stroke × 9 aspect

# Erasure and stroke break parameters (for font generation only)
ERASURE_PERCENT = 0.15  # 15% of white pixels erased
STROKE_BREAK_SIZE = 4   # 4 pixels per break
STROKE_BREAK_COUNT = 6  # 6 breaks scattered randomly


def get_google_fonts(api_key, category=None, num_fonts=None):
    """
    Fetch font list from Google Fonts API.
    
    Args:
        api_key: Google Fonts API key
        category: Font category (handwriting, sans-serif, display, etc.)
        num_fonts: Max number of fonts to return (None = all)
    
    Returns:
        List of font dicts with non-italic variants only
    """
    base_url = "https://www.googleapis.com/webfonts/v1/webfonts"
    
    params = {
        "key": api_key,
        "sort": "popularity"
    }
    
    if category:
        params["category"] = category
    
    print(f"Fetching fonts from Google Fonts API (category: {category or 'all'})...")
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Error: API request failed with status {response.status_code}")
        print(f"Response: {response.text}")
        return []
    
    data = response.json()
    fonts = data.get("items", [])
    
    # Filter fonts: keep only those with non-italic variants
    filtered_fonts = []
    for font in fonts:
        files = font.get("files", {})
        # Get non-italic variants only
        non_italic_files = {k: v for k, v in files.items() if k not in EXCLUDE_VARIANTS}
        
        if non_italic_files:
            font["files"] = non_italic_files
            filtered_fonts.append(font)
    
    total_found = len(filtered_fonts)
    
    if num_fonts is not None:
        filtered_fonts = filtered_fonts[:num_fonts]
    
    print(f"  Found {total_found} fonts with non-italic variants, using {len(filtered_fonts)}")
    return filtered_fonts


def get_all_fonts_by_category(api_key):
    """
    Get fonts according to category limits:
    - handwriting: Top 75
    - sans-serif: Top 20
    - display: Top 10
    
    Returns:
        List of unique fonts with their variants
    """
    all_fonts = []
    seen_families = set()
    
    print("\n" + "="*60)
    print("Fetching fonts by category:")
    print("="*60)
    
    for category, limit in FONT_CATEGORY_LIMITS.items():
        fonts = get_google_fonts(api_key, category=category, num_fonts=limit)
        
        # Add unique fonts only
        for font in fonts:
            family = font["family"]
            if family not in seen_families:
                seen_families.add(family)
                font["_category"] = category  # Track source category
                all_fonts.append(font)
    
    print(f"\nTotal unique fonts: {len(all_fonts)}")
    
    # Summary by category
    by_category = {}
    for font in all_fonts:
        cat = font.get("_category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    
    print("By category:")
    for cat, count in by_category.items():
        print(f"  {cat}: {count}")
    print("="*60 + "\n")
    
    return all_fonts


def download_font(font_url, font_name):
    """
    Download a font file from URL.
    """
    try:
        response = requests.get(font_url, timeout=30)
        
        if response.status_code != 200:
            return None
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf")
        temp_file.write(response.content)
        temp_file.close()
        
        return temp_file.name
        
    except Exception as e:
        print(f"  Error downloading {font_name}: {e}")
        return None


def render_base_digit(digit, font_path, canvas_size=128, font_size=None):
    """
    Render a single digit on a larger canvas for later transformation.
    Returns grayscale image with white digit on black background.
    
    If font_size is None, automatically scales to 50% of canvas height.
    """
    img = Image.new('L', (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)
    
    # Auto-scale font size to 45% of canvas if not specified
    # This ensures no clipping with 1.75x aspect ratio + 30° rotation
    # Worst case: 1.75S × S rotated 30° ≈ 2.0S, so S ≤ canvas/2.0 = 45% is safe
    if font_size is None:
        font_size = int(canvas_size * 0.45)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None
    
    # Get text bounding box and center
    bbox = draw.textbbox((0, 0), digit, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (canvas_size - text_width) // 2 - bbox[0]
    y = (canvas_size - text_height) // 2 - bbox[1]
    
    draw.text((x, y), digit, fill=255, font=font)
    
    return img


def apply_rotation(img, angle):
    """Apply rotation to image."""
    return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)


def apply_shear(img, shear_degrees):
    """Apply vertical shear transformation to image (in degrees)."""
    import math
    width, height = img.size
    # Convert degrees to shear factor
    shear_factor = math.tan(math.radians(shear_degrees))
    # Affine transform matrix for vertical shear (y-axis)
    # [1, 0, 0]
    # [shear, 1, -shear*width/2]
    coeffs = (1, 0, 0,
              shear_factor, 1, -shear_factor * width / 2)
    return img.transform((width, height), Image.Transform.AFFINE, coeffs,
                         resample=Image.Resampling.BILINEAR, fillcolor=0)


def apply_aspect_ratio(img, aspect_factor):
    """Apply aspect ratio distortion (stretch width)."""
    width, height = img.size
    new_width = int(width * aspect_factor)
    stretched = img.resize((new_width, height), Image.Resampling.BILINEAR)
    
    # Center crop/pad back to original size
    result = Image.new('L', (width, height), color=0)
    offset = (width - new_width) // 2
    if offset >= 0:
        result.paste(stretched, (offset, 0))
    else:
        # Crop from center
        crop_offset = -offset
        cropped = stretched.crop((crop_offset, 0, crop_offset + width, height))
        result.paste(cropped, (0, 0))
    
    return result


def find_bounding_box(img):
    """
    Find bounding box of non-zero pixels in image.
    
    Args:
        img: PIL Image (grayscale)
    
    Returns:
        Tuple (x_min, y_min, x_max, y_max) or None if no non-zero pixels
    """
    img_array = np.array(img)
    coords = np.column_stack(np.where(img_array > 0))
    
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return (x_min, y_min, x_max + 1, y_max + 1)


def crop_resize_with_margin(img, target_size, margin=2, bbox=None):
    """
    Crop digit to bounding box, resize to MAXIMIZE digit size while keeping margin,
    and paste into target_size image with exactly margin pixels on all sides.
    
    The digit is scaled to fill as much of the (target_size - 2*margin) area as possible,
    maximizing the digit size while maintaining aspect ratio.
    
    Args:
        img: PIL Image (after augmentation)
        target_size: Final image size (28 or 64)
        margin: Margin in pixels (default 2)
        bbox: Optional pre-computed bounding box (x_min, y_min, x_max, y_max).
              If None, will find bounding box from img.
    
    Returns:
        PIL Image of size target_size x target_size with maximized digit and exactly margin pixels on all sides
    """
    # Initialize debug counter if not exists
    if not hasattr(crop_resize_with_margin, 'debug_count'):
        crop_resize_with_margin.debug_count = 0
    
    # Use provided bbox or find it
    if bbox is None:
        bbox = find_bounding_box(img)
    if bbox is None:
        # No content, return black image of target size
        return Image.new('L', (target_size, target_size), color=0)
    
    x_min, y_min, x_max, y_max = bbox
    
    # DEBUG: Print bounding box info for first 10 images only (SUPPRESSED for full run)
    # if crop_resize_with_margin.debug_count < 10:
    #     print(f"DEBUG crop_resize_with_margin (image {crop_resize_with_margin.debug_count + 1}/10):")
    #     print(f"  Original image size: {img.size}")
    #     print(f"  Bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    
    # Crop to bounding box (no margin yet)
    cropped = img.crop((x_min, y_min, x_max, y_max))
    crop_width = x_max - x_min
    crop_height = y_max - y_min
    
    # if crop_resize_with_margin.debug_count < 10:
    #     print(f"  Crop dimensions: {crop_width} x {crop_height}")
    
    # Calculate target digit size (area to fill)
    digit_size = target_size - 2 * margin
    
    # Scale to maximize digit - use the larger dimension to determine scale
    # This ensures the digit fills as much space as possible while maintaining aspect ratio
    scale = digit_size / max(crop_width, crop_height)
    new_width = int(crop_width * scale)
    new_height = int(crop_height * scale)
    
    # DEBUG: Increment counter (kept for image saving)
    if crop_resize_with_margin.debug_count < 10:
        crop_resize_with_margin.debug_count += 1
        # DEBUG prints suppressed for full run:
        # print(f"  Target digit size: {digit_size} (target_size={target_size}, margin={margin})")
        # print(f"  Scale factor: {scale:.4f}")
        # print(f"  Resized dimensions: {new_width} x {new_height}")
        # print(f"  Final image fill: {new_width}/{digit_size} x {new_height}/{digit_size} = {new_width/digit_size*100:.1f}% x {new_height/digit_size*100:.1f}%")
        # print()
    else:
        crop_resize_with_margin.debug_count += 1
    
    # Resize maintaining aspect ratio
    resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create final image with target_size x target_size (black background)
    result = Image.new('L', (target_size, target_size), color=0)
    
    # Paste resized digit centered in the available area (with margin)
    paste_x = margin + (digit_size - new_width) // 2
    paste_y = margin + (digit_size - new_height) // 2
    result.paste(resized, (paste_x, paste_y))
    
    # DEBUG: Save first 10 images for inspection
    if crop_resize_with_margin.debug_count <= 10:
        import os
        debug_dir = "./data/font_digits_debug"
        os.makedirs(debug_dir, exist_ok=True)
        # Use count for filename (1-indexed, count is already incremented)
        debug_filename = f"{debug_dir}/debug_{crop_resize_with_margin.debug_count:02d}.png"
        result.save(debug_filename)
        if crop_resize_with_margin.debug_count == 10:
            print(f"DEBUG: Saved 10 example images to {debug_dir}/")
    
    return result


def apply_noise(img):
    """Add Gaussian noise to image (matches DigitClassifierALL.py GaussNoise)."""
    img_array = np.array(img, dtype=np.float32) / 255.0
    noise = np.random.normal(0, NOISE_STD, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def apply_stroke_variation(img, variation):
    """
    Apply stroke thickness variation.
    
    Args:
        img: PIL Image
        variation: -1 = thinner (erode with 3x3 cross), 0 = normal, 1 = thicker (dilate)
    
    Returns:
        PIL Image with modified stroke thickness
    """
    if variation == 0:
        return img
    
    img_array = np.array(img)
    
    if variation == -1:
        # Thinning: 3x3 cross-shaped kernel, 1 iteration
        kernel = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], np.uint8)
        eroded = cv2.erode(img_array, kernel, iterations=1)
        return Image.fromarray(eroded)
    else:  # variation == 1
        # Thickening: 2x2 kernel, 1 iteration
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(img_array, kernel, iterations=1)
        return Image.fromarray(dilated)


def apply_random_pixel_erasure(img, erasure_percent=ERASURE_PERCENT):
    """
    Randomly erase a percentage of white stroke pixels.
    
    Args:
        img: PIL Image (grayscale)
        erasure_percent: Fraction of white pixels to erase (default 0.15 = 15%)
    
    Returns:
        PIL Image with some pixels erased
    """
    img_array = np.array(img)
    
    # Find all white pixels (stroke pixels, value > threshold)
    white_mask = img_array > 25  # Threshold for "white"
    white_indices = np.where(white_mask)
    
    if len(white_indices[0]) == 0:
        return img
    
    total_white = len(white_indices[0])
    n_to_erase = int(total_white * erasure_percent)
    
    if n_to_erase == 0:
        return img
    
    # Randomly select pixels to erase
    indices_to_erase = np.random.choice(total_white, n_to_erase, replace=False)
    
    for idx in indices_to_erase:
        y, x = white_indices[0][idx], white_indices[1][idx]
        img_array[y, x] = 0
    
    return Image.fromarray(img_array)


def apply_stroke_breaks(img, break_size=STROKE_BREAK_SIZE, num_breaks=STROKE_BREAK_COUNT):
    """
    Apply small contiguous breaks (gaps) in the stroke.
    
    Args:
        img: PIL Image (grayscale)
        break_size: Size of each break in pixels (default 4)
        num_breaks: Number of breaks to apply (default 6)
    
    Returns:
        PIL Image with stroke breaks
    """
    img_array = np.array(img)
    height, width = img_array.shape
    
    # Find all white pixels (stroke pixels)
    white_mask = img_array > 25
    white_indices = np.where(white_mask)
    
    if len(white_indices[0]) == 0:
        return img
    
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
        img_array[y_start:y_end, x_start:x_end] = 0
    
    return Image.fromarray(img_array)


def generate_augmented_variations(base_img, target_size):
    """
    Generate all augmented variations for a base digit image.
    
    Creates 27 combinations (stroke × aspect) × 4 variation types:
    - 1 rotation (random from either negative or positive range)
    - 1 shear (random from either negative or positive range)
    - 1 with 15% random pixel erasure (no rotation/shear)
    - 1 with 6 stroke breaks at 4px (no rotation/shear)
    = 108 variations
    Plus 1 original = 109 total variations per digit.
    
    Args:
        base_img: Base PIL image (400x400 canvas)
        target_size: Final image size (28 or 64)
    
    Returns:
        List of 109 augmented PIL images (target_size x target_size with exactly 2px margin on all sides)
    """
    variations = []
    
    # Generate all combinations: stroke (3) × aspect (9) = 27
    combinations = []
    for stroke in STROKE_VARIATIONS:
        for aspect in ASPECT_RATIOS:
            combinations.append((stroke, aspect))
    
    # For each combination, generate 4 variation types
    for stroke_var, aspect_ratio in combinations:
        # Variation 1: ROTATION (random from either negative or positive range)
        img = base_img.copy()
        if np.random.rand() < 0.5:
            angle = np.random.uniform(*ROTATION_RANGE_NEG)
        else:
            angle = np.random.uniform(*ROTATION_RANGE_POS)
        img = apply_rotation(img, angle)
        img = apply_aspect_ratio(img, aspect_ratio)
        img = apply_stroke_variation(img, stroke_var)
        # Find bounding box BEFORE noise
        bbox = find_bounding_box(img)
        img = apply_noise(img)
        img = crop_resize_with_margin(img, target_size, margin=2, bbox=bbox)
        variations.append(img)
        
        # Variation 2: SHEAR (random from either negative or positive range)
        img = base_img.copy()
        if np.random.rand() < 0.5:
            shear = np.random.uniform(*SHEAR_RANGE_NEG)
        else:
            shear = np.random.uniform(*SHEAR_RANGE_POS)
        img = apply_shear(img, shear)
        img = apply_aspect_ratio(img, aspect_ratio)
        img = apply_stroke_variation(img, stroke_var)
        # Find bounding box BEFORE noise
        bbox = find_bounding_box(img)
        img = apply_noise(img)
        img = crop_resize_with_margin(img, target_size, margin=2, bbox=bbox)
        variations.append(img)
        
        # Variation 3: 15% RANDOM PIXEL ERASURE (no rotation/shear)
        img = base_img.copy()
        img = apply_aspect_ratio(img, aspect_ratio)
        img = apply_stroke_variation(img, stroke_var)
        img = apply_random_pixel_erasure(img)
        # Find bounding box BEFORE noise
        bbox = find_bounding_box(img)
        img = apply_noise(img)
        img = crop_resize_with_margin(img, target_size, margin=2, bbox=bbox)
        variations.append(img)
        
        # Variation 4: 6 STROKE BREAKS at 4px (no rotation/shear)
        img = base_img.copy()
        img = apply_aspect_ratio(img, aspect_ratio)
        img = apply_stroke_variation(img, stroke_var)
        img = apply_stroke_breaks(img)
        # Find bounding box BEFORE noise
        bbox = find_bounding_box(img)
        img = apply_noise(img)
        img = crop_resize_with_margin(img, target_size, margin=2, bbox=bbox)
        variations.append(img)
    
    # Add 1 original: no rotation, no shear, no erasure, no breaks, stroke=0, aspect=1.0
    img = base_img.copy()
    img = apply_aspect_ratio(img, 1.0)  # Normal aspect
    # No stroke variation (stroke=0, normal)
    # Find bounding box BEFORE noise
    bbox = find_bounding_box(img)
    img = apply_noise(img)
    img = crop_resize_with_margin(img, target_size, margin=2, bbox=bbox)
    variations.append(img)
    
    return variations


def get_font_weights_to_use(files):
    """
    Select ONE font weight to use from available files.
    
    Priority: regular > bold > thin > any available
    
    Returns:
        List of (weight_name, url) tuples (single item)
    """
    # Priority order: regular first, then bold, then thin
    priority_order = ["regular", "400", "500", "700", "600", "800", "900", "300", "200", "100"]
    
    for w in priority_order:
        if w in files:
            return [(w, files[w])]
    
    # If none of the preferred weights found, use whatever is available
    if files:
        first_key = next(iter(files))
        return [(first_key, files[first_key])]
    
    return []


def generate_digit_images(api_key, output_dir, target_size=28):
    """
    Main function to generate augmented digit images from Google Fonts.
    
    Fetches:
    - Top 75 handwriting fonts
    - Top 20 sans-serif fonts
    - Top 10 display fonts
    
    For each font, uses 1 weight (preferring regular).
    Each digit generates 109 variations (27 combinations × 4 + 1 original).
    
    Args:
        api_key: Google Fonts API key
        output_dir: Directory to save generated images
        target_size: Output image size (28 or 64, default: 64)
        
    Returns:
        Tuple of (x_data, y_softmax, y_sigmoid)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_images = []
    all_labels = []
    all_font_families = []  # Track font family for each sample
    all_font_categories = []  # Track font category for each sample
    font_count = 0
    weight_count = 0
    png_files_saved = 0  # Count PNG files saved
    
    # Get fonts by category (75 handwriting, 20 sans-serif, 10 display)
    all_fonts = get_all_fonts_by_category(api_key)
    
    # Calculate expected output
    variations_per_digit = COMBINATIONS_COUNT * 4 + 1  # 27 combinations × 4 + 1 original = 109
    
    # Estimate total: fonts × 1 weight × 10 digits × variations
    est_total = len(all_fonts) * 10 * variations_per_digit
    print(f"Estimated output: ~{est_total:,} images")
    print(f"({len(all_fonts)} fonts × 1 weight × 10 digits × {variations_per_digit} variations)\n")
    
    for font_idx, font in enumerate(all_fonts):
        family = font["family"]
        category = font.get("_category", "unknown")
        files = font.get("files", {})
        
        # Get weights to use (thin, regular, bold)
        weights = get_font_weights_to_use(files)
        
        if not weights:
            continue
        
        font_count += 1
        print(f"[{font_count}/{len(all_fonts)}] {family} ({category}) - {len(weights)} weight(s)")
        
        for weight_name, font_url in weights:
            # Download font
            font_path = download_font(font_url, f"{family}-{weight_name}")
            if not font_path:
                continue
            
            weight_count += 1
            
            try:
                # Generate images for each digit
                for digit in DIGITS:
                    digit_label = int(digit)
                    
                    # Render base digit (400x400) - font size auto-scales to 50% of canvas
                    # Large canvas with smaller font allows room for rotation/shear/aspect without clipping
                    base_img = render_base_digit(digit, font_path, canvas_size=400)
                    if base_img is None:
                        continue
                    
                    # Generate all variations (109 per digit)
                    all_variations = generate_augmented_variations(base_img, target_size=target_size)
                    
                    for var_idx, img in enumerate(all_variations):
                        # Save individual image
                        safe_family = family.replace(" ", "_").replace("/", "_")
                        filename = f"{safe_family}_{weight_name}_d{digit}_v{var_idx:02d}.png"
                        img.save(output_path / filename)
                        png_files_saved += 1
                        
                        # Collect for numpy array
                        img_array = np.array(img, dtype=np.float32) / 255.0
                        all_images.append(img_array)
                        all_labels.append(digit_label)
                        all_font_families.append(family)  # Track font family
                        all_font_categories.append(category)  # Track font category
            
            finally:
                # Clean up temporary font file
                try:
                    os.unlink(font_path)
                except:
                    pass
    
    print(f"\n{'='*60}")
    print(f"Generated {len(all_images):,} digit images")
    print(f"  PNG files saved: {png_files_saved:,}")
    print(f"  Fonts used: {font_count}")
    print(f"  Font weights used: {weight_count}")
    print(f"Images saved to: {output_path}")
    
    # Convert to numpy arrays and split into train/test
    if all_images:
        x_data = np.array(all_images)
        x_data = np.expand_dims(x_data, axis=-1)  # Add channel dimension (N, 64, 64, 1)
        
        # Softmax labels (integer)
        y_softmax = np.array(all_labels, dtype=np.int32)
        
        # Sigmoid labels (one-hot)
        y_sigmoid = np.zeros((len(all_labels), 10), dtype=np.float32)
        for i, label in enumerate(all_labels):
            y_sigmoid[i, label] = 1.0
        
        # Split into train/test (80/20) by digit and font category
        print(f"\n{'='*60}")
        print("Splitting data into train/test sets (80/20)...")
        print("Split strategy: Per digit, per font category, by font family")
        print(f"{'='*60}")
        
        train_indices = []
        test_indices = []
        
        # Split for each digit (0-9)
        for digit in range(10):
            digit_mask = y_softmax == digit
            digit_indices = np.where(digit_mask)[0]
            
            # Group by font category
            for category in ["handwriting", "sans-serif", "display"]:
                category_mask = np.array([all_font_categories[i] == category for i in digit_indices])
                category_indices = digit_indices[category_mask]
                
                if len(category_indices) == 0:
                    continue
                
                # Get unique font families in this category for this digit
                font_families_in_category = [all_font_families[i] for i in category_indices]
                unique_families = list(set(font_families_in_category))
                
                # Shuffle for random split
                np.random.seed(42)  # For reproducibility
                np.random.shuffle(unique_families)
                
                # Split fonts 80/20
                split_idx = int(len(unique_families) * 0.8)
                train_families = set(unique_families[:split_idx])
                test_families = set(unique_families[split_idx:])
                
                # Assign samples to train/test based on font family
                for idx in category_indices:
                    if all_font_families[idx] in train_families:
                        train_indices.append(idx)
                    else:
                        test_indices.append(idx)
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        # Shuffle train and test sets
        np.random.seed(42)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        # Create train/test splits
        x_train = x_data[train_indices]
        y_train_softmax = y_softmax[train_indices]
        y_train_sigmoid = y_sigmoid[train_indices]
        
        x_test = x_data[test_indices]
        y_test_softmax = y_softmax[test_indices]
        y_test_sigmoid = y_sigmoid[test_indices]
        
        # Print split statistics
        print(f"\nTrain set: {len(x_train):,} samples")
        print(f"Test set: {len(x_test):,} samples")
        print(f"Train/Test ratio: {len(x_train)/len(x_test):.2f}:1")
        
        # Print per-digit distribution
        print("\nPer-digit distribution:")
        print("Digit | Train | Test | Total")
        print("-" * 30)
        for digit in range(10):
            train_count = np.sum(y_train_softmax == digit)
            test_count = np.sum(y_test_softmax == digit)
            total_count = train_count + test_count
            print(f"  {digit}   | {train_count:5,} | {test_count:4,} | {total_count:5,}")
        
        # Save train/test splits (include size in filename)
        train_softmax_path = output_path / f"font_digits_train_{target_size}x{target_size}_softmax.npz"
        train_sigmoid_path = output_path / f"font_digits_train_{target_size}x{target_size}_sigmoid.npz"
        test_softmax_path = output_path / f"font_digits_test_{target_size}x{target_size}_softmax.npz"
        test_sigmoid_path = output_path / f"font_digits_test_{target_size}x{target_size}_sigmoid.npz"
        
        np.savez(train_softmax_path, x=x_train, y=y_train_softmax)
        np.savez(train_sigmoid_path, x=x_train, y=y_train_sigmoid)
        np.savez(test_softmax_path, x=x_test, y=y_test_softmax)
        np.savez(test_sigmoid_path, x=x_test, y=y_test_sigmoid)
        
        print(f"\n{'='*60}")
        print("Saved numpy arrays:")
        print(f"  Train Softmax: {train_softmax_path}")
        print(f"    x shape: {x_train.shape}")
        print(f"    y shape: {y_train_softmax.shape} (integer labels 0-9)")
        print(f"  Train Sigmoid: {train_sigmoid_path}")
        print(f"    x shape: {x_train.shape}")
        print(f"    y shape: {y_train_sigmoid.shape} (one-hot labels)")
        print(f"  Test Softmax: {test_softmax_path}")
        print(f"    x shape: {x_test.shape}")
        print(f"    y shape: {y_test_softmax.shape} (integer labels 0-9)")
        print(f"  Test Sigmoid: {test_sigmoid_path}")
        print(f"    x shape: {x_test.shape}")
        print(f"    y shape: {y_test_sigmoid.shape} (one-hot labels)")
        print(f"{'='*60}")
        
        return (x_train, y_train_softmax, y_train_sigmoid), (x_test, y_test_softmax, y_test_sigmoid)
    
    return None, None


def load_font_digits(npz_path):
    """
    Load previously generated font digits.
    
    Args:
        npz_path: Path to the .npz file (softmax or sigmoid version)
        
    Returns:
        Tuple of (x_data, y_data)
    """
    data = np.load(npz_path)
    return data['x'], data['y']


def load_font_digits_train_test(data_dir):
    """
    Load train/test splits of font digits.
    
    Args:
        data_dir: Directory containing the train/test .npz files
        
    Returns:
        Tuple of ((x_train, y_train), (x_test, y_test)) for softmax format
    """
    data_path = Path(data_dir)
    
    train_softmax_path = data_path / "font_digits_train_softmax.npz"
    test_softmax_path = data_path / "font_digits_test_softmax.npz"
    train_sigmoid_path = data_path / "font_digits_train_sigmoid.npz"
    test_sigmoid_path = data_path / "font_digits_test_sigmoid.npz"
    
    if train_softmax_path.exists() and test_softmax_path.exists():
        train_data = np.load(train_softmax_path)
        test_data = np.load(test_softmax_path)
        return (train_data['x'], train_data['y']), (test_data['x'], test_data['y'])
    elif train_sigmoid_path.exists() and test_sigmoid_path.exists():
        train_data = np.load(train_sigmoid_path)
        test_data = np.load(test_sigmoid_path)
        return (train_data['x'], train_data['y']), (test_data['x'], test_data['y'])
    else:
        raise FileNotFoundError(f"Train/test files not found in {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="""
Generate augmented digit images from Google Fonts.

Font selection:
  - Top 75 handwriting fonts by popularity (non-italic)
  - Top 20 sans-serif fonts by popularity (non-italic)  
  - Top 10 display fonts by popularity (non-italic)

Each font uses 1 weight (preferring regular).
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-key", type=str, 
                        default=None,
                        help="Google Fonts API key (or set in code or GOOGLE_FONTS_API_KEY env var)")
    parser.add_argument("--output-dir", type=str, 
                        default="./data/font_digits",
                        help="Output directory for generated images")
    parser.add_argument("--size", type=int,
                        default=28,
                        choices=[28, 64],
                        help="Output image size (28 or 64, default: 28)")
    
    args = parser.parse_args()
    
    # API key priority: command line > code constant > environment variable
    api_key = args.api_key or os.environ.get("GOOGLE_FONTS_API_KEY")
    
    if not api_key:
        print("Error: API key required.")
        print("\nOptions to provide API key:")
        print("1. Pass --api-key YOUR_KEY on command line")
        print("2. Set GOOGLE_FONTS_API_KEY environment variable")
        print("\nTo get an API key:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a project (or select existing)")
        print("3. Enable the 'Google Fonts Developer API'")
        print("4. Go to 'Credentials' and create an API key")
        return
    
    print("="*60)
    print("Font Digit Generator")
    print("="*60)
    print("Configuration:")
    print(f"  Handwriting fonts: Top {HANDWRITING_LIMIT} by popularity (non-italic)")
    print(f"  Sans-serif fonts: Top {SANS_SERIF_LIMIT} by popularity (non-italic)")
    print(f"  Display fonts: Top {DISPLAY_LIMIT} by popularity (non-italic)")
    print(f"  Weights: 1 per font (preferring regular)")
    print(f"  Output size: {args.size}x{args.size}")
    print(f"  Output: {args.output_dir}")
    print("="*60 + "\n")
    
    generate_digit_images(
        api_key=api_key,
        output_dir=args.output_dir,
        target_size=args.size
    )


if __name__ == "__main__":
    main()
