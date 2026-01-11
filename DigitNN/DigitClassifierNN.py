#!/usr/bin/env python3
"""
DigitClassifier.py

Provides functions for creating, training, and using a CNN-based digit classifier.
The classifier is trained on the MNIST and EMNIST Digits datasets to recognize handwritten digits (0-9).
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

try:
    from emnist import extract_training_samples, extract_test_samples
    EMNIST_AVAILABLE = True
except ImportError:
    EMNIST_AVAILABLE = False
    print("Warning: 'emnist' package not available. Install with: pip install emnist")


def create_digit_classifier_model(use_240k_samples=False, use_deep_model=True):
    """
    Create a CNN model for digit classification (0-9).
    
    Args:
        use_240k_samples: Whether to use 240k samples from EMNIST Digits (affects model capacity)
        use_deep_model: Whether to use deep model architecture (default: True)
    
    Returns:
        Compiled Keras model
    """

    # Adjust model capacity based on dataset size
    if use_240k_samples:
        # Larger model for MNIST + EMNIST (more training data)
        number_convolution_channels = 32
        neurons_in_dense_layer = 32
    else:
        # Smaller model for MNIST only
        number_convolution_channels = 32
        neurons_in_dense_layer = 32
        
    if use_deep_model:
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(neurons_in_dense_layer, activation='elu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
        ])
    else:
        # Shallow model architecture (fewer layers)
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(neurons_in_dense_layer, activation='elu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
        ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_augmentation_pipeline(stats_tracker=None):
    """
    Create an albumentations augmentation pipeline for MNIST/EMNIST digit training.
    Includes rotation/slant, image quality issues (blur, noise), and stroke thickness variation.
    
    Note: For MNIST/EMNIST, we do NOT use scale or position/translation transforms
    since digits are already centered and normalized in 28x28 images.
    
    Args:
        stats_tracker: Optional dict to track augmentation statistics
    
    Returns:
        albumentations Compose object
    """
    # Create individual transforms - all can apply together for realistic combinations
    transforms = []
    
    # 1. Rotation OR Slant (mutually exclusive) - no shift, no scale
    # Rotation: rotates entire digit (like tilting paper)
    # Slant: vertical shear for forward/backward tilt
    # These are mutually exclusive since they both affect orientation
    # Note: For MNIST/EMNIST, we do NOT apply scale or position/translation transforms
    #       since digits are already centered and normalized in 28x28 images.
    #       Affine transform defaults to no scale/translate when not specified.
    transforms.append(A.OneOf([
        A.Affine(
            rotate_limit=48,      # 48 degrees max rotation (no scale, no translate)
            p=1.0
        ),
        A.Affine(
            shear={'x': 0, 'y': (-15, 15)},  # Vertical shear for forward/backward tilt (±15 degrees)
            # No scale or translate parameters - defaults to no scaling/translation
            p=1.0
        )
    ], p=0.8))  # 80% chance of getting either rotation OR slant
    
    # 2. Image quality issues - can occur together (realistic for poor scans/photos)
    transforms.append(A.GaussianBlur(blur_limit=(1, 3), p=0.3))  # Light blur
    transforms.append(A.GaussNoise(p=0.2))  # Light noise
    
    # Note: Morphological operations (stroke thickness variation) are applied separately
    # in the data generator, so they can combine with the above augmentations
    
    transform = A.Compose(transforms, p=1.0)
    
    return transform


class ImageDataGeneratorWithAugmentation:
    """
    Custom data generator that applies albumentations augmentations to data.
    """
    def __init__(self, augmentation_pipeline, batch_size=64):
        self.augmentation_pipeline = augmentation_pipeline
        self.batch_size = batch_size
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'augmented_samples': 0,
            'original_samples': 0,
            'morphology_thicker': 0,
            'morphology_thinner': 0,
        }
    
    def flow(self, x, y, batch_size=None):
        """
        Generator that yields batches of augmented data.
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        num_samples = len(x)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        while True:
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]
                
                # Process each image in the batch
                batch_x_aug = []
                for img in batch_x:
                    self.stats['total_samples'] += 1
                    
                    # Convert from (28, 28, 1) to (28, 28) for albumentations
                    img_2d = img.squeeze(axis=-1)
                    
                    # Convert from float [0,1] to uint8 [0,255] for albumentations
                    img_uint8 = (img_2d * 255).astype(np.uint8)
                    
                    # Apply augmentation to 50% of samples (reduce augmented samples by 50%)
                    # All original samples are used, but only 50% get augmented
                    if np.random.random() < 0.5:
                        # Apply augmentation (albumentations expects dict with 'image' key)
                        # Note: All augmentations can apply to the same image simultaneously
                        # based on their individual probabilities
                        self.stats['augmented_samples'] += 1
                        augmented = self.augmentation_pipeline(image=img_uint8)
                        img_aug = augmented['image']
                        
                        # Apply stroke thickness variation using morphological operations
                        # For WHITE digits on BLACK background:
                        # - dilate() makes bright regions (digits) THICKER
                        # - erode() makes bright regions (digits) THINNER
                        if np.random.random() < 0.5:
                            kernel_size = np.random.choice([1, 2])
                            if np.random.random() < 0.5:
                                # Thicker strokes: dilate the white digits
                                self.stats['morphology_thicker'] += 1
                                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                                img_aug = cv2.dilate(img_aug, kernel, iterations=1)
                            #else:
                                # Thinner strokes: erode the white digits
                                #self.stats['morphology_thinner'] += 1
                                #kernel = np.ones((kernel_size, kernel_size), np.uint8)
                                #img_aug = cv2.erode(img_aug, kernel, iterations=1)
                    else:
                        # Keep original image (no augmentation) - 50% of samples remain original
                        self.stats['original_samples'] += 1
                        img_aug = img_uint8
                    
                    # Convert back to float [0,1]
                    img_aug_float = (img_aug.astype(np.float32) / 255.0)
                    
                    # Reshape to (28, 28, 1)
                    img_aug_float = np.expand_dims(img_aug_float, axis=-1)
                    
                    # Ensure values are in [0, 1] range
                    img_aug_float = np.clip(img_aug_float, 0.0, 1.0)
                    
                    batch_x_aug.append(img_aug_float)
                
                batch_x_aug = np.array(batch_x_aug)
                
                yield batch_x_aug, batch_y


class AugmentationStatsCallback(keras.callbacks.Callback):
    """
    Custom callback to print per-epoch augmentation statistics.
    Tracks how many samples were augmented vs original each epoch.
    """
    def __init__(self, datagen, samples_per_epoch):
        self.datagen = datagen
        self.samples_per_epoch = samples_per_epoch
        self.last_total = 0
        self.last_augmented = 0
        self.last_original = 0
    
    def on_epoch_end(self, epoch, logs=None):
        stats = self.datagen.stats
        current_total = stats['total_samples']
        current_augmented = stats['augmented_samples']
        current_original = stats['original_samples']
        
        samples_this_epoch = current_total - self.last_total
        augmented_this_epoch = current_augmented - self.last_augmented
        original_this_epoch = current_original - self.last_original
        
        self.last_total = current_total
        self.last_augmented = current_augmented
        self.last_original = current_original
        
        print(f"\n[Epoch {epoch+1}] Samples processed: {samples_this_epoch} (Augmented: {augmented_this_epoch}, Original: {original_this_epoch})")


def load_and_combine_datasets(use_mnist=True, use_emnist=True):
    """
    Load and combine MNIST and/or EMNIST datasets.
    
    Args:
        use_mnist: Whether to load MNIST dataset (default: True)
        use_emnist: Whether to load EMNIST Digits dataset (default: True)
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) as numpy arrays
        Arrays are normalized to [0, 1] and reshaped to (samples, 28, 28, 1)
    
    Raises:
        ValueError: If no datasets could be loaded
    """
    # Load MNIST if requested
    x_train_mnist = None
    y_train_mnist = None
    x_test_mnist = None
    y_test_mnist = None
    
    if use_mnist:
        print("Loading MNIST dataset...")
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()
        print(f"Loaded MNIST: {len(x_train_mnist)} training, {len(x_test_mnist)} test samples")
    else:
        print("Skipping MNIST dataset (use_mnist=False)")
    
    # Load EMNIST Digits if requested and available
    x_train_emnist = None
    y_train_emnist = None
    x_test_emnist = None
    y_test_emnist = None
    
    if use_emnist and EMNIST_AVAILABLE:
        try:
            print("Loading EMNIST Digits dataset...")
            # Note: 'digits' parameter loads ONLY EMNIST Digits (0-9), not the full EMNIST dataset
            # This gives us ~240,000 additional digit samples beyond MNIST's 60,000
            x_train_emnist, y_train_emnist = extract_training_samples('digits')
            x_test_emnist, y_test_emnist = extract_test_samples('digits')
            print(f"Loaded EMNIST Digits: {len(x_train_emnist)} training, {len(x_test_emnist)} test samples")
            if len(x_train_emnist) == 0:
                print("Warning: EMNIST Digits loaded but has 0 samples. Check dataset installation.")
                x_train_emnist = None
        except Exception as e:
            print(f"Error: Could not load EMNIST Digits: {e}")
            import traceback
            traceback.print_exc()
            x_train_emnist = None
            y_train_emnist = None
            x_test_emnist = None
            y_test_emnist = None
    elif use_emnist and not EMNIST_AVAILABLE:
        print("Skipping EMNIST dataset (EMNIST package not available. Install with: pip install emnist)")
    elif not use_emnist:
        print("Skipping EMNIST dataset (use_emnist=False)")
    
    # Combine datasets based on what's available
    datasets_to_combine = []
    dataset_names = []
    
    if x_train_mnist is not None:
        datasets_to_combine.append((x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist))
        dataset_names.append(f"MNIST ({len(x_train_mnist)} samples)")
    
    if x_train_emnist is not None:
        datasets_to_combine.append((x_train_emnist, y_train_emnist, x_test_emnist, y_test_emnist))
        dataset_names.append(f"EMNIST Digits ({len(x_train_emnist)} samples)")
    
    # Check if we have at least one dataset
    if len(datasets_to_combine) == 0:
        error_msg = "No training data available! "
        if not use_mnist and not use_emnist:
            error_msg += "Both MNIST and EMNIST are disabled."
        elif not use_mnist:
            error_msg += "MNIST is disabled (use_mnist=False) and EMNIST is not available or failed to load."
        elif not use_emnist:
            error_msg += "EMNIST is disabled (use_emnist=False) and MNIST failed to load."
        else:
            error_msg += "Both datasets failed to load."
        raise ValueError(error_msg)
    
    # Combine all available datasets
    if len(datasets_to_combine) == 1:
        x_train, y_train, x_test, y_test = datasets_to_combine[0]
        print(f"Using {dataset_names[0]}: {len(x_train)} training, {len(x_test)} test samples")
    else:
        # Combine multiple datasets
        print(f"Combining datasets: {' + '.join(dataset_names)}")
        x_train = np.concatenate([ds[0] for ds in datasets_to_combine], axis=0)
        y_train = np.concatenate([ds[1] for ds in datasets_to_combine], axis=0)
        x_test = np.concatenate([ds[2] for ds in datasets_to_combine], axis=0)
        y_test = np.concatenate([ds[3] for ds in datasets_to_combine], axis=0)
        print(f"Combined dataset: {len(x_train)} training, {len(x_test)} test samples")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    return x_train, y_train, x_test, y_test


def load_or_create_digit_classifier(classifier_model_path=None, 
train_model=True,
use_augmentation=True, use_mnist=True, use_emnist=True, 
num_epochs=20, use_deep_model=True):
    """
    Load a pre-trained digit classifier or create/train a new one.
    
    Args:
        classifier_model_path: Path to saved classifier model (.keras file)
        use_augmentation: Whether to use data augmentation during training (default: True)
        use_mnist: Whether to include MNIST data for training/validation (default: True)
        use_emnist: Whether to include EMNIST Digits data for training/validation (default: True)
        num_epochs: Number of training epochs (default: 20)
        use_deep_model: Whether to use deep model architecture (default: True)
    
    Returns:
        Trained Keras model for digit classification
    """

    print("===========train_model: ", train_model)
    print("===========classifier_model_path: ", classifier_model_path)
    # Try to load existing model (from specified path only if train_model is False)
    if (not train_model) and os.path.exists(classifier_model_path):
        try:
            print(f"Loading digit classifier from: {classifier_model_path}")
            model = keras.models.load_model(classifier_model_path)
            print("Digit classifier loaded successfully")
            return model
        except Exception as e:
            print(f"Warning: Could not load classifier from {classifier_model_path}: {e}")
            print("Creating new classifier model...")
    
    # We're going to train a new model, so create the run directory now
    # Create timestamped directory for model checkpoints
    base_dir = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data" / "modelForDE"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory: run_YYYY_MM_DD_HH_MM_SS
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model checkpoints will be saved to: {run_dir}")
    
    # Create new model
    print("Creating new digit classifier model...")
    model = create_digit_classifier_model(use_240k_samples=use_emnist, 
    use_deep_model=use_deep_model)
    
    # Try to train on MNIST + EMNIST Digits dataset
    try:
        # Load and combine datasets
        x_train, y_train, x_test, y_test = load_and_combine_datasets(use_mnist=use_mnist, use_emnist=use_emnist)
        
        print(f"Training samples per epoch: {len(x_train)}")
        print(f"Test samples: {len(x_test)}")
        print(f"Number of epochs: {num_epochs}")
        
        # Setup training based on whether augmentation is enabled
        if use_augmentation:
            # Create data augmentation pipeline
            # This will be applied on-the-fly during training
            print("Setting up data augmentation pipeline...")
            augmentation_pipeline = create_augmentation_pipeline()
            
            # Create data generator with augmentation
            train_datagen = ImageDataGeneratorWithAugmentation(
                augmentation_pipeline=augmentation_pipeline,
                batch_size=64
            )
            
            # Train the model with augmented data
            print("Starting training with data augmentation...")
            print("\n" + "="*60)
            print("Augmentation Configuration:")
            print("="*60)
            print("All augmentations can apply to the SAME image simultaneously")
            print("(Based on their individual probabilities)")
            print("\nAugmentation probabilities:")
            print("  50% of samples will be augmented, 50% will remain original")
            print("  Rotation OR Slant (mutually exclusive): 80% (p=0.8) of augmented samples")
            print("    - Rotation: ±48° rotation (no shift, no scale)")
            print("    - Slant: ±15° vertical shear (no shift, no scale)")
            print("  GaussianBlur (image quality): 30% (p=0.3) of augmented samples")
            print("  GaussNoise (image quality): 20% (p=0.2) of augmented samples")
            print("  Morphology - Stroke thickness variation: 50% (p=0.5) of augmented samples")
            print("\nNote: Augmentation is applied on-the-fly - each sample is augmented differently each epoch")
            print(f"Total samples processed over {num_epochs} epochs: ~{len(x_train) * num_epochs}")
            print(f"(~{int(len(x_train) * num_epochs * 0.5)} augmented, ~{int(len(x_train) * num_epochs * 0.5)} original)")
            print("(Each sample is augmented uniquely each time it's seen)")
            print("="*60 + "\n")
            
            stats_callback = AugmentationStatsCallback(train_datagen, len(x_train))
            
            # ModelCheckpoint callback to save model after each epoch with epoch number
            # Save all epoch models in the timestamped run directory
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / "digit_classifier_epoch_{epoch:02d}.keras"),
                save_best_only=False,  # Save every epoch, not just best
                save_weights_only=False,  # Save full model
                verbose=0  # Don't print save messages (already verbose=1 in fit)
            )
            
            print(f"Epoch models will be saved as: {run_dir}/digit_classifier_epoch_XX.keras (one per epoch)")
            
            model.fit(
                train_datagen.flow(x_train, y_train, batch_size=64),
                steps_per_epoch=len(x_train) // 64,
                epochs=num_epochs,
                validation_data=(x_test, y_test),
                verbose=1,
                callbacks=[stats_callback, checkpoint_callback]
            )
            
            # Print final augmentation statistics
            print("\n" + "="*60)
            print("Final Augmentation Statistics:")
            print("="*60)
            stats = train_datagen.stats
            total = stats['total_samples']
            if total > 0:
                print(f"Total samples processed across all epochs: {total}")
                print(f"Average samples per epoch: {total / num_epochs:.0f}")
                print(f"\nAugmentation breakdown:")
                print(f"  Augmented samples: {stats['augmented_samples']}/{total} ({stats['augmented_samples']/total*100:.1f}%)")
                print(f"  Original samples: {stats['original_samples']}/{total} ({stats['original_samples']/total*100:.1f}%)")
                print(f"\nMorphology augmentation application rates (within augmented samples):")
                if stats['augmented_samples'] > 0:
                    print(f"  Morphology - Thicker strokes: {stats['morphology_thicker']}/{stats['augmented_samples']} ({stats['morphology_thicker']/stats['augmented_samples']*100:.1f}% of augmented)")
                    print(f"  Morphology - Thinner strokes: {stats['morphology_thinner']}/{stats['augmented_samples']} ({stats['morphology_thinner']/stats['augmented_samples']*100:.1f}% of augmented)")
                print(f"\nNote: All augmentations (Rotation/Slant, GaussianBlur, GaussNoise, Morphology)")
                print(f"      can apply to the same image simultaneously based on their individual probabilities.")
                print(f"      (Morphology stats are tracked manually; other augmentations are handled by albumentations)")
            print("="*60)
        else:
            # Train the model without augmentation
            print("Starting training WITHOUT data augmentation...")
            print("\n" + "="*60)
            print("Training Configuration:")
            print("="*60)
            print("Training with original data only (no augmentation)")
            print(f"Training samples per epoch: {len(x_train)}")
            print(f"Test samples: {len(x_test)}")
            print("="*60 + "\n")
            
            # ModelCheckpoint callback to save model after each epoch with epoch number
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / "digit_classifier_epoch_{epoch:02d}.keras"),
                save_best_only=False,  # Save every epoch, not just best
                save_weights_only=False,  # Save full model
                verbose=0  # Don't print save messages (already verbose=1 in fit)
            )
            
            print(f"Epoch models will be saved as: {run_dir}/digit_classifier_epoch_XX.keras (one per epoch)")
            
            model.fit(
                x_train, y_train,
                batch_size=64,
                epochs=num_epochs,
                validation_data=(x_test, y_test),
                verbose=1,
                callbacks=[checkpoint_callback]
            )
        
        # Save the final model (also saved by checkpoint, but this ensures final state is saved)
        # Save to run_dir
        final_model_path = str(run_dir / "digit_classifier_final.keras")
        
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        print(f"(Individual epoch models saved in: {run_dir})")
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("Evaluating model on test set...")
        print("="*60)
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        num_test_samples = len(x_test)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4%} ({test_accuracy*num_test_samples:.0f} out of {num_test_samples} test images)")
        
        # Get per-class accuracy
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nPer-class accuracy on test set:")
        print("-" * 40)
        for digit in range(10):
            mask = y_test == digit
            if np.sum(mask) > 0:
                class_accuracy = np.mean(y_pred_classes[mask] == digit)
                correct = np.sum(y_pred_classes[mask] == digit)
                total = np.sum(mask)
                print(f"  Digit {digit}: {class_accuracy:.2%} ({correct}/{total})")
        
        print("="*60)
        print("Digit classifier trained and ready!")
        return model
        
    except Exception as e:
        print(f"Warning: Could not train digit classifier: {e}")
        print("Using untrained model (predictions will be random)")
        return model


def classify_digit(classifier_model, digit_image):
    """
    Classify a single digit image using the CNN model.
    
    Args:
        classifier_model: Trained Keras model
        digit_image: 28x28 greyscale image (numpy array)
    
    Returns:
        Predicted digit (0-9) and confidence score
    """
    # Ensure image is the right shape and type
    if digit_image.shape != (28, 28):
        # Resize if needed
        digit_image = cv2.resize(digit_image, (28, 28))
    
    # Normalize pixel values to [0, 1]
    digit_normalized = digit_image.astype('float32') / 255.0
    
    # The input image should already be in MNIST format: white digits on black background
    # (ensured by BoundingBoxFromYolo.py preprocessing)
    # MNIST: white digits (high values ~1.0) on black background (low values ~0.0)
    
    # Reshape for model input: (1, 28, 28, 1)
    digit_input = digit_normalized.reshape(1, 28, 28, 1)
    
    # Predict
    predictions = classifier_model.predict(digit_input, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_digit])
    
    return predicted_digit, confidence


def main():
    """
    Standalone training function for the digit classifier.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train a digit classifier on MNIST dataset"
    )
    parser.add_argument(
        "-m", "--model-path",
        type=str,
        default=None,
        help="Path to save the trained model (.keras file). Default: ~/.digit_classifier_mnist.keras"
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="True means train model, False means load model"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation during training (augmentation is enabled by default)"
    )
    parser.add_argument(
        "--no-mnist",
        action="store_true",
        help="Exclude MNIST data from training/validation (MNIST is included by default)"
    )
    parser.add_argument(
        "--no-emnist",
        action="store_true",
        help="Exclude EMNIST Digits data from training/validation (EMNIST is included by default)"
    )
    parser.add_argument(
        "--epoch-count",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--no-deep-model",
        action="store_true",
        help="Use shallow model architecture instead of deep model (deep model is used by default)"
    )
    
    args = parser.parse_args()
        
    # Train the model
    print("Starting digit classifier training...")
    use_augmentation = not args.no_augment  # Augmentation is default (True unless --no-augment is set)
    use_mnist = not args.no_mnist  # MNIST is default (True unless --no-mnist is set)
    use_emnist = not args.no_emnist  # EMNIST is default (True unless --no-emnist is set)
    use_deep_model = not args.no_deep_model  # Deep model is default (True unless --no-deep-model is set)
    model = load_or_create_digit_classifier(
        args.model_path, 
        args.train_model,
        use_augmentation=use_augmentation,
        use_mnist=use_mnist,
        use_emnist=use_emnist,
        num_epochs=args.epoch_count,
        use_deep_model=use_deep_model
    )
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

