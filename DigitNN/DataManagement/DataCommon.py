import numpy as np
from pathlib import Path
from PIL import Image

HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"


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