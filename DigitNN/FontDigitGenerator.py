#!/usr/bin/env python3
"""
FontDigitGenerator.py

Downloads fonts from Google Fonts API and generates augmented digit images (0-9)
for use as training data.

Font Selection:
- Top 100 handwriting fonts by popularity (non-italic) - most similar to handwritten digits
- Top 50 sans-serif fonts by popularity (non-italic) - clean digit shapes
- Top 30 display fonts by popularity (non-italic) - varied styles
- 1 weight per font (preferring regular)

Each font generates 360 images:
- 10 digits × 36 variations each
- 18 rotated variations (rotate + noise × stroke × aspect)
- 18 sheared variations (shear + noise × stroke × aspect)

Usage:
    python FontDigitGenerator.py --output-dir ./data/font_digits
    
Output:
    - Individual PNG images
    - font_digits_softmax.npz: x (N, 28, 28, 1), y (N,) integer labels
    - font_digits_sigmoid.npz: x (N, 28, 28, 1), y (N, 10) one-hot labels
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
TARGET_SIZE = (28, 28)  # MNIST format: 28x28 pixels

# Font category limits
HANDWRITING_LIMIT = 100  # Top N handwriting fonts by popularity
SANS_SERIF_LIMIT = 50    # Top N sans-serif fonts by popularity
DISPLAY_LIMIT = 30       # Top N display fonts by popularity

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

# Output format - IDENTICAL TO MNIST:
# - Size: 28x28 pixels
# - Channels: 1 (grayscale)
# - Background: Black (0)
# - Digit: White (normalized to 0-1 range)
# - Dtype: float32
# - Shape: (N, 28, 28, 1)

# Augmentation parameters - MATCHED TO DigitClassifierALL.py
ROTATION_RANGE = (-48, 48)  # degrees (matches DigitClassifierALL.py line 271)
SHEAR_RANGE = (-15, 15)     # degrees vertical shear (matches DigitClassifierALL.py line 272)
NOISE_STD = 0.05            # Gaussian noise std dev (similar to GaussNoise in DigitClassifierALL.py)
STROKE_VARIATIONS = [0, 1]  # 0 = normal, 1 = thicker (dilate)
ASPECT_RATIOS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]  # width stretch factors (0.05 increments)

# Combinations per transform type
# stroke (2) × aspect (9) = 18 combinations per transform type
# Total per digit: 18 rotated + 18 sheared = 36 variations
VARIATIONS_PER_TRANSFORM = 18


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
    - handwriting: ALL
    - sans-serif: Top 100
    - display: Top 50
    
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


def render_base_digit(digit, font_path, canvas_size=64, font_size=48):
    """
    Render a single digit on a larger canvas for later transformation.
    Returns grayscale image with white digit on black background.
    """
    img = Image.new('L', (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)
    
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


def apply_noise(img):
    """Add Gaussian noise to image (matches DigitClassifierALL.py GaussNoise)."""
    img_array = np.array(img, dtype=np.float32) / 255.0
    noise = np.random.normal(0, NOISE_STD, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def apply_stroke_variation(img, variation):
    """Apply stroke thickness variation (0=normal, 1=thicker via dilation)."""
    if variation == 0:
        return img
    
    img_array = np.array(img)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(img_array, kernel, iterations=1)
    return Image.fromarray(dilated)


def generate_augmented_variations(base_img, transform_type='rotate'):
    """
    Generate VARIATIONS_PER_TRANSFORM augmented versions of an image.
    
    Combinations: stroke (2) × aspect (5) = 10 per transform type
    Each variation also gets Gaussian noise applied.
    
    Args:
        base_img: Base PIL image
        transform_type: 'rotate' or 'shear'
    
    Returns:
        List of augmented PIL images
    """
    variations = []
    
    # Generate all combinations: stroke (2) × aspect (5) = 10
    all_combinations = []
    for stroke in STROKE_VARIATIONS:
        for aspect in ASPECT_RATIOS:
            if transform_type == 'rotate':
                angle = np.random.uniform(*ROTATION_RANGE)
                all_combinations.append((angle, stroke, aspect))
            else:  # shear
                shear = np.random.uniform(*SHEAR_RANGE)
                all_combinations.append((shear, stroke, aspect))
    
    # Use all 10 combinations (or sample if we have more)
    selected = all_combinations[:VARIATIONS_PER_TRANSFORM]
    
    for params in selected:
        transform_val, stroke_var, aspect_ratio = params
        
        img = base_img.copy()
        
        # Apply transformations in order
        if transform_type == 'rotate':
            img = apply_rotation(img, transform_val)
        else:
            img = apply_shear(img, transform_val)
        
        img = apply_aspect_ratio(img, aspect_ratio)
        img = apply_stroke_variation(img, stroke_var)
        img = apply_noise(img)  # Gaussian noise always applied
        
        # Resize to target size
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
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


def generate_digit_images(api_key, output_dir):
    """
    Main function to generate augmented digit images from Google Fonts.
    
    Fetches:
    - ALL handwriting fonts
    - Top 100 sans-serif fonts
    - Top 50 display fonts
    
    For each font, uses 1 weight (preferring regular).
    
    Args:
        api_key: Google Fonts API key
        output_dir: Directory to save generated images
        
    Returns:
        Tuple of (x_data, y_softmax, y_sigmoid)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_images = []
    all_labels = []
    font_count = 0
    weight_count = 0
    
    # Get fonts by category (ALL handwriting, top 100 sans-serif, top 50 display)
    all_fonts = get_all_fonts_by_category(api_key)
    
    # Calculate expected output
    variations_per_digit = VARIATIONS_PER_TRANSFORM * 2  # rotate + shear
    
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
                    
                    # Render base digit
                    base_img = render_base_digit(digit, font_path)
                    if base_img is None:
                        continue
                    
                    # Generate rotated variations
                    rotated_variations = generate_augmented_variations(base_img, 'rotate')
                    
                    # Generate sheared variations
                    sheared_variations = generate_augmented_variations(base_img, 'shear')
                    
                    # Combine all variations
                    all_variations = rotated_variations + sheared_variations
                    
                    for var_idx, img in enumerate(all_variations):
                        # Save individual image
                        safe_family = family.replace(" ", "_").replace("/", "_")
                        filename = f"{safe_family}_{weight_name}_d{digit}_v{var_idx:02d}.png"
                        img.save(output_path / filename)
                        
                        # Collect for numpy array
                        img_array = np.array(img, dtype=np.float32) / 255.0
                        all_images.append(img_array)
                        all_labels.append(digit_label)
            
            finally:
                # Clean up temporary font file
                try:
                    os.unlink(font_path)
                except:
                    pass
    
    print(f"\n{'='*60}")
    print(f"Generated {len(all_images):,} digit images")
    print(f"  Fonts used: {font_count}")
    print(f"  Font weights used: {weight_count}")
    print(f"Images saved to: {output_path}")
    
    # Convert to numpy arrays
    if all_images:
        x_data = np.array(all_images)
        x_data = np.expand_dims(x_data, axis=-1)  # Add channel dimension (N, 28, 28, 1)
        
        # Softmax labels (integer)
        y_softmax = np.array(all_labels, dtype=np.int32)
        
        # Sigmoid labels (one-hot)
        y_sigmoid = np.zeros((len(all_labels), 10), dtype=np.float32)
        for i, label in enumerate(all_labels):
            y_sigmoid[i, label] = 1.0
        
        # Save both formats
        softmax_path = output_path / "font_digits_softmax.npz"
        sigmoid_path = output_path / "font_digits_sigmoid.npz"
        
        np.savez(softmax_path, x=x_data, y=y_softmax)
        np.savez(sigmoid_path, x=x_data, y=y_sigmoid)
        
        print(f"\nSaved numpy arrays:")
        print(f"  Softmax: {softmax_path}")
        print(f"    x shape: {x_data.shape}")
        print(f"    y shape: {y_softmax.shape} (integer labels 0-9)")
        print(f"  Sigmoid: {sigmoid_path}")
        print(f"    x shape: {x_data.shape}")
        print(f"    y shape: {y_sigmoid.shape} (one-hot labels)")
        print(f"{'='*60}")
        
        return x_data, y_softmax, y_sigmoid
    
    return None, None, None


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


def main():
    parser = argparse.ArgumentParser(description="""
Generate augmented digit images from Google Fonts.

Font selection:
  - ALL handwriting fonts (non-italic)
  - Top 100 sans-serif fonts by popularity (non-italic)  
  - Top 50 display fonts by popularity (non-italic)

Each font uses 1 weight (preferring regular).
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-key", type=str, 
                        default=None,
                        help="Google Fonts API key (or set in code or GOOGLE_FONTS_API_KEY env var)")
    parser.add_argument("--output-dir", type=str, 
                        default="./data/font_digits",
                        help="Output directory for generated images")
    
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
    print(f"  Output: {args.output_dir}")
    print("="*60 + "\n")
    
    generate_digit_images(
        api_key=api_key,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
