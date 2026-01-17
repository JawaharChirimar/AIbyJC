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
import albumentations as A

# Import augmentation code from classifier
from DigitClassifierNN import create_augmentation_pipeline, ImageDataGeneratorWithAugmentation

try:
    from emnist import extract_training_samples, extract_test_samples
    EMNIST_AVAILABLE = True
except ImportError:
    EMNIST_AVAILABLE = False
    print("Warning: 'emnist' package not available. Install with: pip install emnist")


def create_autoencoder_model():
    """
    Create a convolutional autoencoder model for 28x28 digit images.
    Minimal architecture to force specialization on digit patterns only.
    
    Latent space: 7x7x8 = 392 dimensions (smaller = more selective)
        
    Returns:
        Compiled Keras autoencoder model
    """

    # 1. Input Layer (28x28 grayscale images)
    input_img = keras.Input(shape=(28, 28, 1))

    # 2. ENCODER
    # Compresses: 28x28 -> 14x14 -> 7x7
    # Reduced channels: 8 -> 8 (was 16 -> 32)
    x = layers.Conv2D(8, (3, 3), activation='elu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='elu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # 3. DECODER
    # Reconstructs: 7x7 -> 14x14 -> 28x28
    x = layers.UpSampling2D((2, 2))(encoded)
    x = layers.Conv2D(8, (3, 3), activation='elu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='elu', padding='same')(x)

    # 4. OUTPUT LAYER
    # Sigmoid matches the normalized [0, 1] range of input pixels
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Compile the model
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(
        optimizer='adam', 
        loss='mse',      # Mean Squared Error for reconstruction quality
        metrics=['mae']  # Mean Absolute Error for extra monitoring
    )
      
    return autoencoder


def load_and_combine_datasets():
    """
    Load and combine MNIST and EMNIST datasets for autoencoder training.
        
    Returns:
        Tuple of (x_train, x_test) as numpy arrays
        Arrays are normalized to [0, 1] and reshaped to (samples, 28, 28, 1)
    
    Raises:
        ValueError: If no datasets could be loaded
    """
    # Load MNIST if requested
    x_train_mnist = None
    x_test_mnist = None
    
    print("Loading MNIST dataset...")
    (x_train_mnist, _), (x_test_mnist, _) = keras.datasets.mnist.load_data()
    print(f"Loaded MNIST: {len(x_train_mnist)} training, {len(x_test_mnist)} test samples")
    
    # Load EMNIST Digits if requested and available
    x_train_emnist = None
    x_test_emnist = None
    
    if EMNIST_AVAILABLE:
        try:
            print("Loading EMNIST Digits dataset...")
            x_train_emnist, _ = extract_training_samples('digits')
            x_test_emnist, _ = extract_test_samples('digits')
            print(f"Loaded EMNIST Digits: {len(x_train_emnist)} training, {len(x_test_emnist)} test samples")
        except Exception as e:
            print(f"Error: Could not load EMNIST Digits: {e}")
            x_train_emnist = None
            x_test_emnist = None
    
    # Combine datasets
    datasets_to_combine = []
    dataset_names = []
    
    if x_train_mnist is not None:
        datasets_to_combine.append((x_train_mnist, x_test_mnist))
        dataset_names.append(f"MNIST ({len(x_train_mnist)} samples)")
    
    if x_train_emnist is not None:
        datasets_to_combine.append((x_train_emnist, x_test_emnist))
        dataset_names.append(f"EMNIST Digits ({len(x_train_emnist)} samples)")
    
    if len(datasets_to_combine) == 0:
        raise ValueError("No training data available!")
    
    # Combine all available datasets
    if len(datasets_to_combine) == 1:
        x_train, x_test = datasets_to_combine[0]
        print(f"Using {dataset_names[0]}: {len(x_train)} training, {len(x_test)} test samples")
    else:
        print(f"Combining datasets: {' + '.join(dataset_names)}")
        x_train = np.concatenate([ds[0] for ds in datasets_to_combine], axis=0)
        x_test = np.concatenate([ds[1] for ds in datasets_to_combine], axis=0)
        print(f"Combined dataset: {len(x_train)} training, {len(x_test)} test samples")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for autoencoder (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    return x_train, x_test


def train_autoencoder(num_epochs=20):
    """
    Train an autoencoder on MNIST/EMNIST datasets.
    
    Args:
        num_epochs: Number of training epochs (default: 20)
    
    Returns:
        Trained autoencoder model
    """
    # Create timestamped directory for model
    base_dir = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data" / "autoencoder"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Autoencoder checkpoints will be saved to: {run_dir}")
        
    # Create model
    print("Creating autoencoder model...")
    autoencoder= create_autoencoder_model()
    
    # Load datasets
    x_train, x_test = load_and_combine_datasets()
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Model architecture: {'Shallow (2 conv layers)'}")
    print(f"Data augmentation: {'Enabled'}")
    
    # ModelCheckpoint callback
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(run_dir / "autoencoder_epoch_{epoch:02d}.keras"),
        save_best_only=False,
        save_weights_only=False,
        verbose=0
    )
    
    print(f"Epoch models will be saved as: {run_dir}/autoencoder_epoch_XX.keras")
    
    print("\n" + "="*60)
    print("Setting up data augmentation...")
    print("="*60)
    print("50% of samples will be augmented, 50% will remain original")
    print("Each augmented sample produces 2 images: one rotated, one sheared")
    print("="*60 + "\n")
    
    # Create augmentation pipeline (for blur/noise)
    augmentation_pipeline = create_augmentation_pipeline()
    
    # Create data generator with augmentation (reuse from classifier)
    # For autoencoder, we use dummy labels since generator expects (x, y)
    dummy_y_train = np.zeros(len(x_train), dtype=np.int32)
    dummy_y_test = np.zeros(len(x_test), dtype=np.int32)
    batch_size = 64
    
    train_datagen = ImageDataGeneratorWithAugmentation(
        augmentation_pipeline=augmentation_pipeline,
        batch_size=batch_size
    )
        
    # Create wrapper generator for autoencoder (input = target)
    def autoencoder_generator():
        for x_batch, y_batch in train_datagen.flow(x_train, dummy_y_train, batch_size=batch_size):
            # For autoencoder: input = target (both are the same images)
            yield x_batch, x_batch
    
    # Train with augmented data
    print("Starting training with data augmentation...")
    autoencoder.fit(
        autoencoder_generator(),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=num_epochs,
        validation_data=(x_test, x_test),  # No augmentation on validation
        verbose=1,
        callbacks=[checkpoint_callback]
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
        digit_image: 28x28 greyscale image (numpy array, uint8 0-255 or float 0-1)
        threshold: MSE threshold (if None, uses 95th percentile from training)
    
    Returns:
        Tuple of (is_digit: bool, mse: float)
    """
    # Ensure image is the right shape and type
    if digit_image.shape != (28, 28):
        digit_image = cv2.resize(digit_image, (28, 28))
    
    # Normalize to [0, 1] if needed
    if digit_image.dtype == np.uint8:
        digit_normalized = digit_image.astype('float32') / 255.0
    else:
        digit_normalized = digit_image.astype('float32')
        if digit_normalized.max() > 1.0:
            digit_normalized = digit_normalized / 255.0
    
    # Reshape for model input: (1, 28, 28, 1)
    digit_input = digit_normalized.reshape(1, 28, 28, 1)
    
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
