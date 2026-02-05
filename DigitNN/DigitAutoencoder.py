#!/usr/bin/env python3
"""
DigitAutoencoder.py

Provides functions for creating, training, and using an autoencoder for digit validation.
The autoencoder is trained on MNIST and EMNIST Digits datasets to learn digit patterns.
It can be used to detect if an image is likely a digit by measuring reconstruction error.
"""

import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import pre-generated augmented data loader
from DataManagement.NonDigitGenerator import DATA_DIR
from DataManagement.PregenAugmentedData import load_augmented_data


class AutoencoderDiagnosticsCallback(keras.callbacks.Callback):
    """
    Callback to print per-epoch diagnostics for autoencoder.
    Shows reconstruction error statistics per digit class and overall.
    """
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        # Precompute per-class masks for digits 0-9
        self.class_masks = {}
        for digit in range(10):
            mask = y_val == digit
            self.class_masks[digit] = mask
    
    def on_epoch_end(self, epoch, logs=None):
        # Get reconstructions
        reconstructions = self.model.predict(self.x_val, verbose=0)
        
        # Calculate MSE per sample
        mse_per_sample = np.mean((self.x_val - reconstructions) ** 2, axis=(1, 2, 3))
        
        # Overall statistics
        overall_mse = np.mean(mse_per_sample)
        overall_std = np.std(mse_per_sample)
        overall_min = np.min(mse_per_sample)
        overall_max = np.max(mse_per_sample)
        overall_p95 = np.percentile(mse_per_sample, 95)
        
        results = [f"MSE: {overall_mse:.6f}", f"Std: {overall_std:.6f}", 
                   f"Min: {overall_min:.6f}", f"Max: {overall_max:.6f}", 
                   f"P95: {overall_p95:.6f}"]
        
        # Per-digit reconstruction error
        per_digit_results = []
        for digit in range(10):
            mask = self.class_masks[digit]
            if np.sum(mask) > 0:
                digit_mse = np.mean(mse_per_sample[mask])
                digit_count = np.sum(mask)
                per_digit_results.append(f"{digit}: {digit_mse:.6f} ({digit_count:,})")
        
        # Print diagnostics
        print(f"  [Autoencoder] {' | '.join(results)}")
        print(f"  [Per-digit MSE] {' | '.join(per_digit_results)}")


def create_autoencoder_model():
    """
    Create a convolutional autoencoder model for 64x64 digit images.
    
    Encoder: Conv32 → Pool → Conv64 → Pool → Conv32 → Pool
    Bottleneck: 8x8x32 = 2,048 dimensions (2:1 compression from 4,096 pixels)
    Decoder: UpSample → Conv64 → UpSample → Conv32 → UpSample → Conv32 → Conv1
        
    Returns:
        Compiled Keras autoencoder model
    """

    # 1. Input Layer (64x64 grayscale images)
    input_img = keras.Input(shape=(64, 64, 1))

    # 2. ENCODER
    # 64x64x1 -> 64x64x32 -> 32x32x32 -> 32x32x64 -> 16x16x64 -> 16x16x32 -> 8x8x32
    x = layers.Conv2D(32, (3, 3), activation='elu', padding='same')(input_img)  # 64x64x32
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 32x32x32
    x = layers.Conv2D(64, (3, 3), activation='elu', padding='same')(x)  # 32x32x64
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 16x16x64
    x = layers.Conv2D(32, (3, 3), activation='elu', padding='same')(x)  # 16x16x32
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)  # 8x8x32 (bottleneck: 2,048 dims)

    # 3. DECODER
    # 8x8x32 -> 16x16x32 -> 16x16x64 -> 32x32x64 -> 32x32x32 -> 64x64x32 -> 64x64x1
    x = layers.UpSampling2D((2, 2))(encoded)  # 16x16x32
    x = layers.Conv2D(64, (3, 3), activation='elu', padding='same')(x)  # 16x16x64
    x = layers.UpSampling2D((2, 2))(x)  # 32x32x64
    x = layers.Conv2D(32, (3, 3), activation='elu', padding='same')(x)  # 32x32x32
    x = layers.UpSampling2D((2, 2))(x)  # 64x64x32
    x = layers.Conv2D(32, (3, 3), activation='elu', padding='same')(x)  # 64x64x32

    # 4. OUTPUT LAYER
    # Sigmoid matches the normalized [0, 1] range of input pixels
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # 64x64x1

    # Compile the model
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(
        optimizer='adam', 
        loss='mse',      # Mean Squared Error for reconstruction quality
        metrics=['mae']  # Mean Absolute Error for extra monitoring
    )
      
    return autoencoder


def load_augmented_datasets(image_size=64):
    """
    Load pre-generated, pre-augmented, balanced data for autoencoder training.
    Uses the same data loader as DigitClassifierSoftMax11.py.
    Loads from EMNIST, USPS, ARDIS, and Google Fonts (digits only, no non-digits).
        
    Args:
        image_size: Image size (64, default: 64)
        
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) as numpy arrays
        - x_train, x_test: Arrays normalized to [0, 1] and shaped (samples, image_size, image_size, 1)
        - y_train, y_test: Labels (0-9 for digits only)
    
    Raises:
        ValueError: If no datasets could be loaded
    """
    print(f"Loading pre-generated augmented data ({image_size}x{image_size})...")
    print("Note: Autoencoder uses ONLY digits (0-9), non-digits will be filtered out")
    x_train_all, y_train_all, x_test, y_test = load_augmented_data(image_size=image_size)
    
    if x_train_all is None:
        raise ValueError(f"Failed to load augmented data. Run augmentation scripts for size {image_size}")
    
    # For autoencoder, we only need images (input = target)
    # Filter to only use digit classes (0-9), exclude non-digits (class 10)
    digit_mask_train = y_train_all < 10
    digit_mask_test = y_test < 10
    
    n_non_digits_train = np.sum(y_train_all == 10)
    n_non_digits_test = np.sum(y_test == 10)
    
    x_train_digits = x_train_all[digit_mask_train]
    y_train_digits = y_train_all[digit_mask_train]
    x_test_digits = x_test[digit_mask_test]
    y_test_digits = y_test[digit_mask_test]
    
    print(f"\n✓ Filtered out non-digits (autoencoder uses digits only):")
    print(f"  Excluded from training: {n_non_digits_train:,} non-digit samples")
    print(f"  Excluded from test: {n_non_digits_test:,} non-digit samples")
    print(f"\nLoaded digit images for autoencoder:")
    print(f"  Training: {len(x_train_digits):,} digit samples")
    print(f"  Test: {len(x_test_digits):,} digit samples")
    
    return x_train_digits, y_train_digits, x_test_digits, y_test_digits


def train_autoencoder(num_epochs=20):
    """
    Train an autoencoder on pre-generated augmented digit datasets.
    Uses the same data as DigitClassifierSoftMax11.py: EMNIST, USPS, ARDIS, Google Fonts.
    Only uses digit classes (0-9), excludes non-digits (class 10).
    
    Args:
        num_epochs: Number of training epochs (default: 20)
    
    Returns:
        Trained autoencoder model (64x64 input)
    """
    # Create timestamped directory for model
    base_dir = DATA_DIR / "autoencoder"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Autoencoder checkpoints will be saved to: {run_dir}")
        
    # Create model
    print("Creating autoencoder model...")
    autoencoder = create_autoencoder_model()
    
    # Load pre-generated augmented datasets (64x64)
    x_train, y_train, x_test, y_test = load_augmented_datasets(image_size=64)
    
    print(f"\nTraining samples: {len(x_train):,}")
    print(f"Test samples: {len(x_test):,}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Model architecture: 64x64 input, encoder (Conv32→Pool→Conv64→Pool→Conv32→Pool), bottleneck (8x8x32=2,048), decoder (UpSample→Conv64→UpSample→Conv32→UpSample→Conv32→Conv1)")
    print(f"Data: Pre-generated augmented data (EMNIST, USPS, ARDIS, Google Fonts)")
    
    # ModelCheckpoint callback
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(run_dir / "autoencoder_epoch_{epoch:02d}.keras"),
        save_best_only=False,
        save_weights_only=False,
        verbose=0
    )
    
    print(f"Epoch models will be saved as: {run_dir}/autoencoder_epoch_XX.keras")
    
    # Train with pre-generated augmented data
    batch_size = 128
    print("\n" + "="*60)
    print("Starting training with pre-generated augmented data...")
    print("="*60)
    
    # Training dataset - use from_tensor_slices (references data, no copy)
    # For autoencoder: input = target, so we use (x_train, x_train)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset - use from_tensor_slices (references data, no copy)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test))
    val_dataset = val_dataset.batch(batch_size)
    
    # Diagnostics callback
    diagnostics_callback = AutoencoderDiagnosticsCallback(x_test, y_test)
    
    # Early stopping - stop when validation loss stops improving
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,  # Stop if no improvement for 5 epochs
        min_delta=0.0001,  # Minimum change to qualify as improvement
        restore_best_weights=True,  # Restore best weights when stopping
        verbose=1
    )
    
    autoencoder.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        verbose=1,
        callbacks=[checkpoint_callback, diagnostics_callback, early_stopping]
    )
    
    # Save final model
    autoencoder_model_path = str(run_dir / "autoencoder_final.keras")
    autoencoder.save(autoencoder_model_path)
    print(f"Final autoencoder saved to: {autoencoder_model_path}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating autoencoder on test set...")
    print("="*60)
    test_loss, test_mse = autoencoder.evaluate(x_test, x_test, verbose=0)
    print(f"\nTest Loss (MSE): {test_loss:.4f}")
    print(f"Test MSE (metric): {test_mse:.4f}")
    
    # Calculate reconstruction error statistics
    reconstructions = autoencoder.predict(x_test, verbose=0)
    mse_per_sample = np.mean((x_test - reconstructions) ** 2, axis=(1, 2, 3))
    print(f"\nReconstruction error statistics on test set:")
    print(f"  Mean MSE: {np.mean(mse_per_sample):.6f}")
    print(f"  Std MSE: {np.std(mse_per_sample):.6f}")
    print(f"  Min MSE: {np.min(mse_per_sample):.6f}")
    print(f"  Max MSE: {np.max(mse_per_sample):.6f}")
    print(f"  95th percentile MSE: {np.percentile(mse_per_sample, 95):.6f}")
    print("="*60)
    
    return autoencoder


def load_autoencoder(autoencoder_model_path):
    """
    Load a trained autoencoder model.
    
    Args:
        autoencoder_model_path: Path to saved autoencoder model (.keras file)
    
    Returns:
        Loaded Keras autoencoder model
    """
    if not os.path.exists(autoencoder_model_path):
        raise FileNotFoundError(f"Autoencoder model not found: {autoencoder_model_path}")
    
    print(f"Loading autoencoder from: {autoencoder_model_path}")
    autoencoder = keras.models.load_model(autoencoder_model_path)
    print("Autoencoder loaded successfully")
    return autoencoder


def is_likely_digit(autoencoder, digit_image, threshold=None):
    """
    Check if an image is likely a digit by measuring reconstruction error.
    
    Args:
        autoencoder: Trained autoencoder model
        digit_image: 64x64 greyscale image (numpy array, uint8 0-255 or float 0-1)
        threshold: MSE threshold (if None, uses 95th percentile from training)
    
    Returns:
        Tuple of (is_digit: bool, mse: float)
    """
    # Ensure image is the right shape and type
    if digit_image.shape != (64, 64):
        digit_image = cv2.resize(digit_image, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize to [0, 1] if needed
    if digit_image.dtype == np.uint8:
        digit_normalized = digit_image.astype('float32') / 255.0
    else:
        digit_normalized = digit_image.astype('float32')
        if digit_normalized.max() > 1.0:
            digit_normalized = digit_normalized / 255.0
    
    # Reshape for model input: (1, 64, 64, 1)
    digit_input = digit_normalized.reshape(1, 64, 64, 1)
    
    # Reconstruct
    reconstruction = autoencoder.predict(digit_input, verbose=0)
    reconstruction_output = reconstruction[0, :, :, 0]
    
    # Debug: Check value ranges
    # print(f"Input range: [{digit_normalized.min():.3f}, {digit_normalized.max():.3f}]")
    # print(f"Reconstruction range: [{reconstruction_output.min():.3f}, {reconstruction_output.max():.3f}]")
    
    # Calculate MSE
    mse = np.mean((digit_normalized - reconstruction_output) ** 2)
    
    # Default threshold: 0.008 (can be adjusted based on training statistics)
    # Low MSE = good reconstruction = likely a digit
    if threshold is None:
        threshold = 0.008
    
    is_digit = mse < threshold
    
    return is_digit, float(mse)


def test_image_file(autoencoder_model_path, image_path, threshold=None):
    """
    Test a single image file to see if it's likely a digit.
    
    Args:
        autoencoder_model_path: Path to trained autoencoder model
        image_path: Path to image file to test
        threshold: MSE threshold (None = use default)
    
    Returns:
        Tuple of (is_digit: bool, mse: float)
    """
    autoencoder = load_autoencoder(autoencoder_model_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    is_digit, mse = is_likely_digit(autoencoder, image, threshold=threshold)
    return is_digit, mse


def test_directory(autoencoder_model_path, directory_path, threshold=None, 
                  image_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Test all image files in a directory.
    
    Args:
        autoencoder_model_path: Path to trained autoencoder model
        directory_path: Path to directory containing images
        threshold: MSE threshold (None = use default)
        image_extensions: Tuple of file extensions to test
    
    Returns:
        List of tuples: [(filepath, is_digit, mse), ...]
    """
    autoencoder = load_autoencoder(autoencoder_model_path)
    directory = Path(directory_path)
    
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory_path}")
    
    results = []
    image_files = []
    
    # Find all image files
    for ext in image_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return results
    
    print(f"Found {len(image_files)} image files. Testing...")
    
    for image_file in sorted(image_files):
        try:
            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not load {image_file.name}")
                continue
            
            is_digit, mse = is_likely_digit(autoencoder, image, threshold=threshold)
            results.append((str(image_file), is_digit, mse))
            
            status = "✓ DIGIT" if is_digit else "✗ NOT DIGIT"
            print(f"{image_file.name:30s} | {status:15s} | MSE: {mse:.6f}")
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            results.append((str(image_file), None, None))
    
    return results


def main():
    """
    Standalone function for training or testing with autoencoder.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train an autoencoder or test images for digit validation"
    )
    
    # Subcommands: train or test
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train an autoencoder')
    # Train command arguments
    train_parser.add_argument(
        "--epoch-count",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test image(s) for digit validation')
    test_parser.add_argument(
        "-m", "--model-path",
        type=str,
        required=True,
        help="Path to trained autoencoder model (.keras file)"
    )
    test_parser.add_argument(
        "-i", "--image",
        type=str,
        help="Path to single image file to test"
    )
    test_parser.add_argument(
        "-d", "--directory",
        type=str,
        help="Path to directory containing images to test"
    )
    test_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help="MSE threshold (default: 0.01, lower = stricter)"
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Starting autoencoder training...")
        autoencoder = train_autoencoder(
            num_epochs=args.epoch_count
        )
        print("\nTraining complete!")
    
    elif args.command == 'test':
        if not args.image and not args.directory:
            test_parser.error("Must provide either --image or --directory")
        
        if args.image:
            # Test single image
            print(f"Testing image: {args.image}")
            is_digit, mse = test_image_file(args.model_path, args.image, threshold=args.threshold)
            status = "✓ LIKELY A DIGIT" if is_digit else "✗ NOT LIKELY A DIGIT"
            print(f"\nResult: {status}")
            print(f"MSE: {mse:.6f}")
            if args.threshold:
                print(f"Threshold: {args.threshold}")
        
        if args.directory:
            # Test directory
            print(f"\nTesting images in directory: {args.directory}")
            print("="*70)
            results = test_directory(args.model_path, args.directory, threshold=args.threshold)
            print("="*70)
            
            # Summary
            total = len(results)
            digits = sum(1 for _, is_digit, _ in results if is_digit)
            not_digits = sum(1 for _, is_digit, _ in results if is_digit is False)
            errors = sum(1 for _, is_digit, _ in results if is_digit is None)
            
            print(f"\nSummary:")
            print(f"  Total files: {total}")
            print(f"  Likely digits: {digits}")
            print(f"  Not likely digits: {not_digits}")
            if errors > 0:
                print(f"  Errors: {errors}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
