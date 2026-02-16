import numpy as np
from pathlib import Path
from PIL import Image

HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"

MARGIN_SIZE = 4  # Margin in pixels for cropping/resizing


def float_to_uint8(img_array):
    """
    Convert float32 image array [0, 1] to uint8 [0, 255] with proper rounding.

    Args:
        img_array: numpy array with float values in [0, 1]

    Returns:
        numpy array with uint8 values in [0, 255]
    """
    return np.clip(img_array * 255.0, 0, 255).round().astype(np.uint8)


def upscale_images_to_size(images, target_size):
    """
    Upscale batch of images to target_size x target_size with proper centering.

    Finds bounding box of each digit, crops, and resizes with proper margins
    to ensure digits are centered and not cut off.

    Args:
        images: numpy array (N, H, W) uint8
        target_size: Target size (28 or 64)

    Returns:
        numpy array (N, target_size, target_size) uint8
    """
    upscaled = []
    ctr=0
    for img in images:
        pil_img = Image.fromarray(img)
        # Use crop_resize_with_margin to properly center and add margins
        centered_img = crop_resize_with_margin(pil_img, target_size, idx=None)
        upscaled.append(np.array(centered_img))
        ctr+=1

    return np.array(upscaled, dtype=np.uint8)


def find_bounding_box(img,idx=None):
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

    coords_orig = np.column_stack(np.where(img_array >= 0))
    y0_min, x0_min = coords_orig.min(axis=0)
    y0_max, x0_max = coords_orig.max(axis=0)

    BUFFER = min(x_min-x0_min, y_min-y0_min, x0_max-x_max, y0_max-y_max)
    BUFFER = min(BUFFER, 2)

    if idx is not None and idx==20301:
        print(f"BB:Original bbox for idx {idx}: ({x_min}, {y_min}, {x_max}, {y_max})")
        print(f"BB:Image size: {img.size}")
        img.save(f"data/MNIST/debug_bbox_20301.png")

    x_min = max(x_min - BUFFER, x0_min)
    y_min = max(y_min - BUFFER, y0_min)
    x_max = min(x_max + BUFFER, x0_max)
    y_max = min(y_max + BUFFER, y0_max)

    return (x_min, y_min, x_max, y_max)


def crop_resize_with_margin(img, target_size, bbox=None, idx=None):
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
        bbox = find_bounding_box(img, idx=idx)

    if bbox is None:
        # No content, return black image of target size
        return Image.new('L', (target_size, target_size), color=0)
    
    x_min, y_min, x_max, y_max = bbox

    if idx is not None and idx==20301:
        print(f"CR: Original bbox for idx {idx}: ({x_min}, {y_min}, {x_max}, {y_max})")
    
    # Crop to bounding box (no margin yet)
    # Note: crop box is (left, upper, right, lower) and right/lower are exclusive, so add 1 to include max pixel
    cropped = img.crop((x_min, y_min, x_max+1, y_max+1))
    
    if idx is not None and idx==20301:
        cropped.save(f"data/MNIST/debug_cropped_bbox_20301.png")

    crop_width = x_max - x_min
    crop_height = y_max - y_min
    
    # Calculate target digit size (area to fill)
    digit_size = target_size - 2 * margin
    
    # Scale to maximize digit - use the larger dimension to determine scale
    # This ensures the digit fills as much space as possible while maintaining aspect ratio
    scale = digit_size / max(crop_width, crop_height)
    scale = scale*1.0
    new_width = int(crop_width * scale)
    new_height = int(crop_height * scale)
    
    # Resize maintaining aspect ratio
    #resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized = cropped.resize((new_width, new_height), Image.Resampling.BILINEAR)
    
    # Create final image with target_size x target_size (black background)
    result = Image.new('L', (target_size, target_size), color=0)
    
    # Paste resized digit centered in the available area (with margin)
    paste_x = margin + (digit_size - new_width + 1) // 2
    paste_y = margin + (digit_size - new_height + 1) // 2
    result.paste(resized, (paste_x, paste_y))
    
    return result
