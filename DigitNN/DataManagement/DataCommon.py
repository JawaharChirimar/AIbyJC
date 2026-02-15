import numpy as np
from pathlib import Path
from PIL import Image

HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"

MARGIN_SIZE = 2  # Margin in pixels for cropping/resizing


def upscale_images_to_size(images, target_size):
    """
    Upscale batch of images to target_size x target_size using LANCZOS.
    
    Args:
        images: numpy array (N, H, W) uint8
        target_size: Target size (28 or 64)
    
    Returns:
        numpy array (N, target_size, target_size) uint8
    """
    upscaled = []
    for img in images:
        pil_img = Image.fromarray(img)
        upscaled_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        upscaled.append(np.array(upscaled_img))

    return np.array(upscaled, dtype=np.uint8)


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


def crop_resize_with_margin(img, target_size, bbox=None):
    """
    Crop digit to bounding box, resize to MAXIMIZE digit size while keeping margin,
    and paste into target_size image with exactly margin pixels on all sides.
    
    The digit is scaled to fill as much of the (target_size - 2*margin) area as possible,
    maximizing the digit size while maintaining aspect ratio.
    
    Args:
        img: PIL Image (after augmentation)
        target_size: Final image size (28 or 64)
        bbox: Optional pre-computed bounding box (x_min, y_min, x_max, y_max).
              If None, will find bounding box from img.
    
    Returns:
        PIL Image of size target_size x target_size with maximized digit and 
        exactly MARGIN_SIZE pixels on all sides
    """

    margin = MARGIN_SIZE

    # Use provided bbox or find it
    if bbox is None:
        bbox = find_bounding_box(img)

    if bbox is None:
        # No content, return black image of target size
        return Image.new('L', (target_size, target_size), color=0)
    
    x_min, y_min, x_max, y_max = bbox
    
    # Crop to bounding box (no margin yet)
    cropped = img.crop((x_min, y_min, x_max, y_max))
    crop_width = x_max - x_min
    crop_height = y_max - y_min
    
    # Calculate target digit size (area to fill)
    digit_size = target_size - 2 * margin
    
    # Scale to maximize digit - use the larger dimension to determine scale
    # This ensures the digit fills as much space as possible while maintaining aspect ratio
    scale = digit_size / max(crop_width, crop_height)
    new_width = int(crop_width * scale)
    new_height = int(crop_height * scale)
    
    # Resize maintaining aspect ratio
    resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create final image with target_size x target_size (black background)
    result = Image.new('L', (target_size, target_size), color=0)
    
    # Paste resized digit centered in the available area (with margin)
    paste_x = margin + (digit_size - new_width) // 2
    paste_y = margin + (digit_size - new_height) // 2
    result.paste(resized, (paste_x, paste_y))
    
    return result
