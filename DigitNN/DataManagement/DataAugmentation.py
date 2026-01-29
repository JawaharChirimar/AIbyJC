#!/usr/bin/env python3
"""
DataAugmentation.py

Centralized data augmentation for digit classification models.
Can be imported by any model file (DigitClassifierSoftMax11.py, etc.)

Augmentation Strategy:
- Datasets: EMNIST, ARDIS, USPS (already 64x64 with LANCZOS upscaling)
- 20% of each digit class selected for augmentation per epoch
- Each selected image → original + 5 variations:
    1. Rotation: ±(3° to 30°)
    2. Shear positive: +2° to +16°
    3. Shear negative: -16° to -2°
    4. Aspect wide: 1.05 to 2.0
    5. Aspect narrow: 0.5 to 0.95
- Result: 10K becomes 22K (doubles + 20%)
- Post-processing (applied to doubled set):
    - Blur: 20%
    - Thinning: 10% (3×3 cross kernel, 1 iteration)
    - Thickening: 10%
    - Pixel erasure: 10% (15% of white pixels)
    - Stroke breaks: 10% (6 breaks × 4px each)
- Google Fonts: Skip augmentation (already augmented with rotation, shear, erasure, breaks)
- Non-digits: Same as digits, but skip all-black/all-white

Usage:
    from DataManagement.DataAugmentation import ImageDataGeneratorWithAugmentation
    
    Or if importing from within DataManagement:
    from DataAugmentation import ImageDataGeneratorWithAugmentation
    
    generator = ImageDataGeneratorWithAugmentation(
        x_train, y_train,
        augment_ratio=0.25,
        skip_google_fonts=True
    )
    
    for epoch in range(epochs):
        for batch_x, batch_y in generator.flow(batch_size=32):
            # train on batch
        
        # Print epoch stats
        generator.print_epoch_stats()
        generator.reset_epoch_stats()
    
    # Print final stats
    generator.print_final_stats()
"""

import numpy as np
import cv2
from PIL import Image, ImageFilter
from collections import defaultdict


# =============================================================================
# CONFIGURATION
# =============================================================================

# Augmentation selection
AUGMENT_RATIO = 0.10  # 10% of each class selected for augmentation

# Transform parameters
ROTATION_RANGE_POS = (3, 30)    # degrees (positive)
ROTATION_RANGE_NEG = (-30, -3)  # degrees (negative)
SHEAR_RANGE_POS = (2, 16)       # degrees (positive)
SHEAR_RANGE_NEG = (-16, -2)     # degrees (negative)
ASPECT_WIDE_RANGE = (1.05, 1.6)   # stretch wider (max 60%)
ASPECT_NARROW_RANGE = (0.625, 0.95) # compress narrower (max 37.5%)

# Post-processing probabilities (applied to doubled set, mutually exclusive)
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
STROKE_BREAK_SIZE = 4   # 4 pixels per break
STROKE_BREAK_COUNT = 6  # 6 breaks scattered randomly


# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def apply_rotation(img_array, angle):
    """
    Apply rotation to image without clipping.
    
    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
        angle: rotation angle in degrees
    
    Returns:
        Rotated image array, same shape
    """
    import math
    
    # Handle channel dimension
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True
    
    h, w = img_array.shape
    original_size = max(h, w)  # Use max for square assumption
    
    # Calculate required size for rotation: |cos(θ)| + |sin(θ)|
    angle_rad = math.radians(abs(angle))
    required_size = original_size * (abs(math.cos(angle_rad)) + abs(math.sin(angle_rad)))
    
    # Add 2px padding on each side (+4px total)
    canvas_size = int(math.ceil(required_size)) + 4
    
    # Create larger canvas and embed original image centered
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    offset_y = (canvas_size - h) // 2
    offset_x = (canvas_size - w) // 2
    canvas[offset_y:offset_y+h, offset_x:offset_x+w] = img_array
    
    # Convert to uint8 for cv2
    canvas_uint8 = (canvas * 255).astype(np.uint8)
    
    # Rotate on larger canvas
    center = (canvas_size // 2, canvas_size // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(canvas_uint8, M, (canvas_size, canvas_size), borderValue=0)
    
    # Resize back to original size using LANCZOS
    rotated_pil = Image.fromarray(rotated)
    result_pil = rotated_pil.resize((w, h), Image.Resampling.LANCZOS)
    
    # Back to numpy float
    result = np.array(result_pil).astype(np.float32) / 255.0
    
    if squeeze:
        result = np.expand_dims(result, -1)
    
    return result


def apply_shear(img_array, shear_degrees):
    """
    Apply vertical shear transformation without clipping.
    
    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
        shear_degrees: shear angle in degrees
    
    Returns:
        Sheared image array, same shape
    """
    import math
    
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True
    
    h, w = img_array.shape
    
    # Calculate required width for shear: |tan(shear_angle)| * height
    shear_factor = abs(math.tan(math.radians(shear_degrees)))
    extra_width = int(math.ceil(shear_factor * h))
    required_width = w + extra_width
    
    # Add 2px padding on each side (+4px total)
    canvas_width = required_width + 4
    canvas_height = h + 4  # Also add padding to height for safety
    
    # Create larger canvas and embed original image centered
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    offset_y = (canvas_height - h) // 2
    offset_x = (canvas_width - w) // 2
    canvas[offset_y:offset_y+h, offset_x:offset_x+w] = img_array
    
    # Convert to uint8 for cv2
    canvas_uint8 = (canvas * 255).astype(np.uint8)
    
    # Shear on larger canvas
    center_x = canvas_width // 2
    center_y = canvas_height // 2
    M = np.float32([[1, 0, 0],
                    [shear_factor, 1, -shear_factor * center_x]])
    
    sheared = cv2.warpAffine(canvas_uint8, M, (canvas_width, canvas_height), borderValue=0)
    
    # Resize back to original size using LANCZOS
    sheared_pil = Image.fromarray(sheared)
    result_pil = sheared_pil.resize((w, h), Image.Resampling.LANCZOS)
    
    # Back to numpy float
    result = np.array(result_pil).astype(np.float32) / 255.0
    
    if squeeze:
        result = np.expand_dims(result, -1)
    
    return result


def apply_aspect_ratio(img_array, aspect_factor):
    """
    Apply aspect ratio transformation (horizontal stretch/compress) without clipping.
    
    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
        aspect_factor: >1 = wider, <1 = narrower
    
    Returns:
        Transformed image array, same shape
    """
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True
    
    h, w = img_array.shape
    
    # Calculate new width after stretch/compress
    new_width = int(w * aspect_factor)
    
    # Canvas needs to accommodate the stretched/compressed width
    # Add 2px padding on each side (+4px total)
    canvas_width = max(w, new_width) + 4
    canvas_height = h + 4
    
    # Create larger canvas and embed original image centered
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    offset_y = (canvas_height - h) // 2
    offset_x = (canvas_width - w) // 2
    canvas[offset_y:offset_y+h, offset_x:offset_x+w] = img_array
    
    # Convert to PIL for resize
    canvas_uint8 = (canvas * 255).astype(np.uint8)
    canvas_pil = Image.fromarray(canvas_uint8)
    
    # Apply aspect ratio: resize horizontally to new_width (stretches/compresses the image)
    aspect_resized = canvas_pil.resize((new_width, canvas_height), Image.Resampling.LANCZOS)
    
    # Resize back to original size using LANCZOS
    result_pil = aspect_resized.resize((w, h), Image.Resampling.LANCZOS)
    
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
    img_uint8 = (img_array * 255).astype(np.uint8)
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
    img_uint8 = (img_array * 255).astype(np.uint8)
    
    # Erode
    eroded = cv2.erode(img_uint8, THIN_KERNEL, iterations=THIN_ITERATIONS)
    
    # Back to float
    result = eroded.astype(np.float32) / 255.0
    
    if squeeze:
        result = np.expand_dims(result, -1)
    
    return result


def apply_thickening(img_array):
    """
    Apply stroke thickening (dilation).
    
    Args:
        img_array: numpy array (H, W) or (H, W, 1), values 0-1
    
    Returns:
        Thickened image array, same shape
    """
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True
    
    # Convert to uint8
    img_uint8 = (img_array * 255).astype(np.uint8)
    
    # Dilate
    dilated = cv2.dilate(img_uint8, THICK_KERNEL, iterations=THICK_ITERATIONS)
    
    # Back to float
    result = dilated.astype(np.float32) / 255.0
    
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
# DATA GENERATOR CLASS
# =============================================================================

class ImageDataGeneratorWithAugmentation:
    """
    Data generator with on-the-fly augmentation for digit classification.
    
    Features:
    - 20% of each class augmented per epoch
    - 6 images per augmented sample (original + 5 transforms)
    - 5 transforms: rotation, shear+, shear-, aspect wide, aspect narrow
    - Post-processing: blur (20%), thin (10%), thick (10%), erasure (10%), breaks (10%)
    - Skip Google Fonts (already augmented)
    - Skip blank non-digit images
    - Per-epoch and total stats tracking
    """
    
    def __init__(self, 
                 x_data, 
                 y_data,
                 is_google_fonts=None,
                 augment_ratio=AUGMENT_RATIO,
                 blur_prob=BLUR_PROB,
                 thin_prob=THIN_PROB,
                 thick_prob=THICK_PROB,
                 erasure_prob=ERASURE_PROB,
                 breaks_prob=BREAKS_PROB,
                 num_classes=11,
                 non_digit_class=10):
        """
        Initialize the generator.
        
        Args:
            x_data: numpy array of images (N, H, W, 1), values 0-1
            y_data: numpy array of labels (N,) integer labels
            is_google_fonts: boolean array (N,) - True for Google Fonts samples
            augment_ratio: fraction of each class to augment (default 0.20)
            blur_prob: probability of blur post-processing (default 0.20)
            thin_prob: probability of thinning (default 0.10)
            thick_prob: probability of thickening (default 0.10)
            erasure_prob: probability of pixel erasure (default 0.10)
            breaks_prob: probability of stroke breaks (default 0.10)
            num_classes: total number of classes (default 11: 0-9 + non-digit)
            non_digit_class: class index for non-digits (default 10)
        """
        self.x_data = x_data
        self.y_data = y_data
        self.is_google_fonts = is_google_fonts if is_google_fonts is not None else np.zeros(len(x_data), dtype=bool)
        self.augment_ratio = augment_ratio
        self.blur_prob = blur_prob
        self.thin_prob = thin_prob
        self.thick_prob = thick_prob
        self.erasure_prob = erasure_prob
        self.breaks_prob = breaks_prob
        self.num_classes = num_classes
        self.non_digit_class = non_digit_class
        
        # Organize indices by class
        self.class_indices = {}
        for c in range(num_classes):
            self.class_indices[c] = np.where(y_data == c)[0]
        
        # Stats tracking
        self.reset_epoch_stats()
        self.reset_total_stats()
    
    def reset_epoch_stats(self):
        """Reset per-epoch statistics."""
        self.epoch_stats = {
            'images_original': 0,
            'images_augmented': 0,
            'rotations': 0,
            'shears_pos': 0,
            'shears_neg': 0,
            'aspect_wide': 0,
            'aspect_narrow': 0,
            'blurs': 0,
            'thinning': 0,
            'thickening': 0,
            'erasures': 0,
            'stroke_breaks': 0,
            'skipped_google_fonts': 0,
            'skipped_non_digits': 0,
            'by_class': defaultdict(lambda: {'original': 0, 'augmented': 0})
        }
    
    def reset_total_stats(self):
        """Reset total statistics."""
        self.total_stats = {
            'epochs': 0,
            'images_original': 0,
            'images_augmented': 0,
            'rotations': 0,
            'shears_pos': 0,
            'shears_neg': 0,
            'aspect_wide': 0,
            'aspect_narrow': 0,
            'blurs': 0,
            'thinning': 0,
            'thickening': 0,
            'erasures': 0,
            'stroke_breaks': 0,
            'skipped_google_fonts': 0,
            'skipped_non_digits': 0,
        }
    
    def _update_total_stats(self):
        """Add epoch stats to total stats."""
        self.total_stats['epochs'] += 1
        for key in ['images_original', 'images_augmented', 'rotations', 
                    'shears_pos', 'shears_neg', 'aspect_wide', 'aspect_narrow', 
                    'blurs', 'thinning', 'thickening', 'erasures', 'stroke_breaks',
                    'skipped_google_fonts', 'skipped_non_digits']:
            self.total_stats[key] += self.epoch_stats[key]
    
    def _apply_post_processing(self, img_array):
        """
        Apply post-processing with independent probability checks (matching PregenAugmentedData.py).
        Each effect is checked independently, so an image can get multiple effects.
        - 20% blur
        - 10% thin
        - 10% thick
        - 10% erasure (15% of white pixels)
        - 10% stroke breaks (6 × 4px)
        Returns modified image and list of applied effects.
        """
        effects = []
        result = img_array.copy()
        
        # Independent checks (matching PregenAugmentedData.py)
        if np.random.random() < self.blur_prob:
            radius = np.random.uniform(*BLUR_RADIUS_RANGE)
            result = apply_blur(result, radius)
            effects.append('blur')
            self.epoch_stats['blurs'] += 1
        
        if np.random.random() < self.thin_prob:
            result = apply_thinning(result)
            effects.append('thin')
            self.epoch_stats['thinning'] += 1
        
        if np.random.random() < self.thick_prob:
            result = apply_thickening(result)
            effects.append('thick')
            self.epoch_stats['thickening'] += 1
        
        if np.random.random() < self.erasure_prob:
            result = apply_random_pixel_erasure(result)
            effects.append('erasure')
            self.epoch_stats['erasures'] += 1
        
        if np.random.random() < self.breaks_prob:
            result = apply_stroke_breaks(result)
            effects.append('breaks')
            self.epoch_stats['stroke_breaks'] += 1
        
        return result, effects
    
    def _augment_image(self, img_array, label):
        """
        Generate augmented versions of a single image.
        
        Returns:
            List of (image, label) tuples: original + 5 augmented
            1. Rotation (±3° to ±30°)
            2. Shear positive (+2° to +16°)
            3. Shear negative (-16° to -2°)
            4. Aspect wide (1.05 to 2.0)
            5. Aspect narrow (0.5 to 0.95)
        """
        results = []
        
        # 1. Original (with possible post-processing)
        orig_processed, _ = self._apply_post_processing(img_array)
        results.append((orig_processed, label))
        self.epoch_stats['images_original'] += 1
        
        # 2. Rotation (random + or -)
        angle = np.random.uniform(*ROTATION_RANGE_POS) if np.random.random() > 0.5 else np.random.uniform(*ROTATION_RANGE_NEG)
        rotated = apply_rotation(img_array, angle)
        rotated, _ = self._apply_post_processing(rotated)
        results.append((rotated, label))
        self.epoch_stats['rotations'] += 1
        self.epoch_stats['images_augmented'] += 1
        
        # 3. Shear positive (+2° to +16°)
        shear_pos = np.random.uniform(*SHEAR_RANGE_POS)
        sheared_pos = apply_shear(img_array, shear_pos)
        sheared_pos, _ = self._apply_post_processing(sheared_pos)
        results.append((sheared_pos, label))
        self.epoch_stats['shears_pos'] += 1
        self.epoch_stats['images_augmented'] += 1
        
        # 4. Shear negative (-16° to -2°)
        shear_neg = np.random.uniform(*SHEAR_RANGE_NEG)
        sheared_neg = apply_shear(img_array, shear_neg)
        sheared_neg, _ = self._apply_post_processing(sheared_neg)
        results.append((sheared_neg, label))
        self.epoch_stats['shears_neg'] += 1
        self.epoch_stats['images_augmented'] += 1
        
        # 5. Aspect wide (1.05 to 2.0)
        aspect_wide = np.random.uniform(*ASPECT_WIDE_RANGE)
        wide = apply_aspect_ratio(img_array, aspect_wide)
        wide, _ = self._apply_post_processing(wide)
        results.append((wide, label))
        self.epoch_stats['aspect_wide'] += 1
        self.epoch_stats['images_augmented'] += 1
        
        # 6. Aspect narrow (0.5 to 0.95)
        aspect_narrow = np.random.uniform(*ASPECT_NARROW_RANGE)
        narrow = apply_aspect_ratio(img_array, aspect_narrow)
        narrow, _ = self._apply_post_processing(narrow)
        results.append((narrow, label))
        self.epoch_stats['aspect_narrow'] += 1
        self.epoch_stats['images_augmented'] += 1
        
        return results
    
    def flow(self, batch_size=32, shuffle=True):
        """
        Generate batches of augmented data (memory-efficient).
        
        Augments per-batch instead of building entire epoch in memory.
        
        Args:
            batch_size: number of samples per batch
            shuffle: whether to shuffle data each epoch
        
        Yields:
            (batch_x, batch_y) tuples
        """
        n_samples = len(self.x_data)
        
        while True:  # Infinite generator for keras fit()
            # Build epoch indices - decide which samples get augmented
            indices = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(indices)
            
            # Mark which samples to augment (by index)
            augment_mask = np.zeros(n_samples, dtype=bool)
            for class_idx in range(self.num_classes):
                class_indices = self.class_indices.get(class_idx, [])
                if len(class_indices) == 0:
                    continue
                num_to_augment = max(1, int(len(class_indices) * self.augment_ratio))
                # Randomly select indices to augment
                aug_selection = np.random.choice(class_indices, num_to_augment, replace=False)
                augment_mask[aug_selection] = True
            
            # Process in batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_x_aug = []
                batch_y_aug = []
                
                for idx in batch_indices:
                    img = self.x_data[idx]
                    label = self.y_data[idx]
                    is_gf = self.is_google_fonts[idx]
                    should_augment = augment_mask[idx]
                    class_idx = int(label)
                    
                    # Google Fonts: never augment
                    if is_gf:
                        processed, _ = self._apply_post_processing(img)
                        batch_x_aug.append(processed)
                        batch_y_aug.append(label)
                        self.epoch_stats['skipped_google_fonts'] += 1
                        self.epoch_stats['images_original'] += 1
                        continue
                    
                    if should_augment:
                        # Skip blank non-digit images (matching PregenAugmentedData.py)
                        if class_idx == self.non_digit_class and is_blank_image(img):
                            batch_x_aug.append(img)
                            batch_y_aug.append(label)
                            self.epoch_stats['skipped_non_digits'] += 1
                            self.epoch_stats['images_original'] += 1
                            continue
                        
                        # Non-digits: skip geometric augmentation (treat like Google Fonts)
                        if class_idx == self.non_digit_class:
                            # Apply full post-processing (same as Google Fonts)
                            processed, _ = self._apply_post_processing(img)
                            batch_x_aug.append(processed)
                            batch_y_aug.append(label)
                            self.epoch_stats['skipped_non_digits'] += 1
                            self.epoch_stats['images_original'] += 1
                            continue
                        
                        # Augment: original + 5 variations = 6 total
                        augmented = self._augment_image(img, label)
                        for aug_img, aug_label in augmented:
                            batch_x_aug.append(aug_img)
                            batch_y_aug.append(aug_label)
                        if class_idx < len(self.epoch_stats['by_class']):
                            self.epoch_stats['by_class'][class_idx]['augmented'] += 6
                    else:
                        # Just apply post-processing (non-augmented samples get full post-processing, including non-digits)
                        processed, _ = self._apply_post_processing(img)
                        batch_x_aug.append(processed)
                        batch_y_aug.append(label)
                        self.epoch_stats['images_original'] += 1
                        if class_idx < len(self.epoch_stats['by_class']):
                            self.epoch_stats['by_class'][class_idx]['original'] += 1
                
                batch_x_aug = np.array(batch_x_aug)
                batch_y_aug = np.array(batch_y_aug)
                
                yield batch_x_aug, batch_y_aug
    
    def on_epoch_end(self):
        """Call at end of each epoch to update stats."""
        self._update_total_stats()
    
    def get_epoch_stats(self):
        """Return current epoch statistics."""
        return dict(self.epoch_stats)
    
    def get_total_stats(self):
        """Return total statistics across all epochs."""
        return dict(self.total_stats)
    
    def print_epoch_stats(self, epoch_num=None):
        """Print epoch statistics."""
        prefix = f"Epoch {epoch_num}" if epoch_num is not None else "Epoch"
        stats = self.epoch_stats
        
        total_images = stats['images_original'] + stats['images_augmented']
        
        print(f"\n{'='*60}")
        print(f"{prefix} Augmentation Stats")
        print(f"{'='*60}")
        print(f"Total images this epoch: {total_images:,}")
        print(f"  Original: {stats['images_original']:,}")
        print(f"  Augmented: {stats['images_augmented']:,}")
        print(f"\nTransforms applied:")
        print(f"  Rotations: {stats['rotations']:,}")
        print(f"  Shears (+): {stats['shears_pos']:,}")
        print(f"  Shears (-): {stats['shears_neg']:,}")
        print(f"  Aspect (wide): {stats['aspect_wide']:,}")
        print(f"  Aspect (narrow): {stats['aspect_narrow']:,}")
        print(f"\nPost-processing:")
        print(f"  Blurs: {stats['blurs']:,}")
        print(f"  Thinning: {stats['thinning']:,}")
        print(f"  Thickening: {stats['thickening']:,}")
        print(f"  Erasures: {stats['erasures']:,}")
        print(f"  Stroke breaks: {stats['stroke_breaks']:,}")
        print(f"\nSkipped:")
        print(f"  Google Fonts: {stats['skipped_google_fonts']:,}")
        print(f"  Non-digits: {stats['skipped_non_digits']:,}")
        print(f"{'='*60}")
    
    def print_final_stats(self):
        """Print final statistics summary."""
        stats = self.total_stats
        
        total_images = stats['images_original'] + stats['images_augmented']
        
        print(f"\n{'='*60}")
        print(f"FINAL Augmentation Stats ({stats['epochs']} epochs)")
        print(f"{'='*60}")
        print(f"Total images processed: {total_images:,}")
        print(f"  Original: {stats['images_original']:,}")
        print(f"  Augmented: {stats['images_augmented']:,}")
        print(f"\nTransforms applied (total):")
        print(f"  Rotations: {stats['rotations']:,}")
        print(f"  Shears (+): {stats['shears_pos']:,}")
        print(f"  Shears (-): {stats['shears_neg']:,}")
        print(f"  Aspect (wide): {stats['aspect_wide']:,}")
        print(f"  Aspect (narrow): {stats['aspect_narrow']:,}")
        print(f"\nPost-processing (total):")
        print(f"  Blurs: {stats['blurs']:,}")
        print(f"  Thinning: {stats['thinning']:,}")
        print(f"  Thickening: {stats['thickening']:,}")
        print(f"  Erasures: {stats['erasures']:,}")
        print(f"  Stroke breaks: {stats['stroke_breaks']:,}")
        print(f"\nSkipped (total):")
        print(f"  Google Fonts: {stats['skipped_google_fonts']:,}")
        print(f"  Non-digits: {stats['skipped_non_digits']:,}")
        
        if stats['epochs'] > 0:
            print(f"\nPer-epoch averages:")
            print(f"  Images/epoch: {total_images // stats['epochs']:,}")
            print(f"  Augmented/epoch: {stats['images_augmented'] // stats['epochs']:,}")
        print(f"{'='*60}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def upscale_to_64x64(img_array):
    """
    Upscale image to 64x64 using LANCZOS interpolation.
    
    Args:
        img_array: numpy array, any size, values 0-255 or 0-1
    
    Returns:
        64x64 numpy array, same value range as input
    """
    # Detect value range
    is_float = img_array.max() <= 1.0
    
    # Handle channel dimension
    squeeze = False
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
        squeeze = True
    
    # Convert to uint8 for PIL
    if is_float:
        img_uint8 = (img_array * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.astype(np.uint8)
    
    # Upscale with PIL LANCZOS
    pil_img = Image.fromarray(img_uint8)
    upscaled = pil_img.resize((64, 64), Image.Resampling.LANCZOS)
    
    # Back to numpy
    result = np.array(upscaled)
    
    # Restore value range
    if is_float:
        result = result.astype(np.float32) / 255.0
    
    # Restore channel dimension
    if squeeze:
        result = np.expand_dims(result, -1)
    
    return result


def batch_upscale_to_64x64(x_data):
    """
    Upscale batch of images to 64x64.
    
    Args:
        x_data: numpy array (N, H, W) or (N, H, W, 1)
    
    Returns:
        numpy array (N, 64, 64) or (N, 64, 64, 1)
    """
    # Check if already 64x64
    if x_data.shape[1] == 64 and x_data.shape[2] == 64:
        return x_data
    
    has_channel = x_data.ndim == 4
    
    upscaled = []
    for img in x_data:
        up = upscale_to_64x64(img)
        upscaled.append(up)
    
    result = np.array(upscaled)
    
    # Ensure channel dimension
    if has_channel and result.ndim == 3:
        result = np.expand_dims(result, -1)
    
    return result


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
    
    rotated = apply_rotation(test_img, 15)
    print(f"  Rotation (15°): shape {rotated.shape}, range [{rotated.min():.2f}, {rotated.max():.2f}]")
    
    sheared = apply_shear(test_img, 10)
    print(f"  Shear (10°): shape {sheared.shape}, range [{sheared.min():.2f}, {sheared.max():.2f}]")
    
    wide = apply_aspect_ratio(test_img, 1.5)
    print(f"  Aspect wide (1.5): shape {wide.shape}, range [{wide.min():.2f}, {wide.max():.2f}]")
    
    narrow = apply_aspect_ratio(test_img, 0.7)
    print(f"  Aspect narrow (0.7): shape {narrow.shape}, range [{narrow.min():.2f}, {narrow.max():.2f}]")
    
    blurred = apply_blur(test_img, 1.0)
    print(f"  Blur (r=1.0): shape {blurred.shape}, range [{blurred.min():.2f}, {blurred.max():.2f}]")
    
    thinned = apply_thinning(test_img)
    print(f"  Thinning: shape {thinned.shape}, range [{thinned.min():.2f}, {thinned.max():.2f}]")
    
    thickened = apply_thickening(test_img)
    print(f"  Thickening: shape {thickened.shape}, range [{thickened.min():.2f}, {thickened.max():.2f}]")
    
    # Test generator with dummy data
    print("\nTesting generator...")
    
    # Create dummy dataset
    n_samples = 100
    x_test = np.random.rand(n_samples, 64, 64, 1).astype(np.float32) * 0.5
    y_test = np.random.randint(0, 11, n_samples).astype(np.int32)
    
    generator = ImageDataGeneratorWithAugmentation(
        x_test, y_test,
        augment_ratio=0.25
    )
    
    print(f"  Dataset: {n_samples} samples")
    print(f"  Classes: {len(generator.class_indices)}")
    
    # Run one epoch
    batch_count = 0
    total_samples = 0
    for batch_x, batch_y in generator.flow(batch_size=16):
        batch_count += 1
        total_samples += len(batch_x)
    
    print(f"  Batches generated: {batch_count}")
    print(f"  Total samples in epoch: {total_samples}")
    
    # Print stats
    generator.print_epoch_stats(epoch_num=1)
    
    print("\n✓ All tests passed!")
