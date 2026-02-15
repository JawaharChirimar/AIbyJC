#!/usr/bin/env python3
"""
DigitClassifierSoftMax10.py

Provides functions for creating, training, and using a CNN-based digit classifier.
Uses softmax with 10 classes: digits 0-9 only (no non-digit class).

Uses pre-generated augmented data from the npz files.
"""

import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from DigitClassifierHelper import (BalancedLoss, load_digit_classifier, 
    create_logit_model, calibrate_energy_scorer_helper,
    classify_digit)
from OODDetection import (EnergyScorer, _build_logit_model, 
    _is_softmax_model, _is_logit_model, _build_energy_model_from_path, 
    LogitLayer)    

# Import augmentation module
from StratifiedBatchGenerator import create_stratified_batch_generator

# Data directory
HOME_PATH = Path.home()
if "ubuntu" in str(HOME_PATH).lower():
    DATA_DIR = Path.home() / "AIbyJC" / "DigitNN" / "data"
else:
    DATA_DIR = Path.home() / "Development" / "AIbyJC" / "DigitNN" / "data"

# =============================================================================
# CONFIGURABLE CONSTANTS
# =============================================================================

BATCH_SIZE = 64             # Batch size for training


def create_digit_classifier_model(input_size, use_balanced_loss, 
lambda_weight, learning_rate, neurons_in_dense_layer, use_combined):
    """
    Create a CNN model for digit classification with 10 classes (digits 0-9 only).
    
    Uses deep model architecture with 3 or 4 conv layers, optimized for input_size x input_size images.
    Always uses softmax activation.
    
    Args:
        input_size: Image size (28 or 64)
        use_balanced_loss: Use BalancedLoss instead of cross-entropy
        lambda_weight: Variance penalty weight for BalancedLoss
        learning_rate: Learning rate for Adam optimizer
        neurons_in_dense_layer: Number of neurons in final dense layer
        use_combined: if true 4 conv layers, if false 3 conv layers
    
    Returns:
        Compiled Keras model
    """
    # Model capacity for large dataset (90K to 120K samples)
    number_convolution_channels = 32
    number_convolution_channelsF = 64
    dropout_rate = 0.25
        
    # Choose loss function
    if use_balanced_loss:
        loss_function = BalancedLoss(num_classes=10, var_penalty=lambda_weight)
    else:
        loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    accuracy_metric = 'accuracy'
    
    # Deep model architecture (3 conv layers) for input_size x input_size input
    #conv(number_convolution_channels) → BN → pool(2,2) → 
    #conv(number_convolution_channelsF) → BN → pool(2,2) → dropout(0.25) → 
    #conv(number_convolution_channelsF) → BN → pool(2,2) → dropout(0.25) → 
    #flatten → dense(neurons_in_dense_layer) → BN → dropout(0.5) → dense(10) 
    model_layers = [
        layers.Input(shape=(input_size, input_size, 1)),
    ]

    if use_combined:
        model_layers.extend([
            layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
            layers.BatchNormalization(),
        ])

    model_layers.extend([
        layers.Conv2D(number_convolution_channels, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(number_convolution_channelsF, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        layers.Conv2D(number_convolution_channelsF, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)), 
        layers.Dropout(dropout_rate),
        layers.Flatten(),
        layers.Dense(neurons_in_dense_layer, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(10)  # 10 classes: digits 0-9 only
    ])

    model = keras.Sequential(model_layers)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=[accuracy_metric]
    )
    
    return model



class Softmax10DiagnosticsCallback(keras.callbacks.Callback):
    """
    Callback to print per-epoch diagnostics for softmax 10-class mode.
    Shows per-digit accuracy (0-9) and computes balanced score metric.
    """
    def __init__(self, x_val, y_val, lambda_weight):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.lambda_weight = lambda_weight  # Weight for variance penalty
        # Precompute per-class masks
        self.class_masks = {}
        for digit in range(10):
            mask = y_val == digit
            self.class_masks[digit] = mask
    
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        y_pred = self.model.predict(self.x_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        results = []
        per_class_accuracies = []
        
        # Per-digit accuracy (classes 0-9)
        for digit in range(10):
            mask = self.class_masks[digit]
            if np.sum(mask) > 0:
                digit_acc = np.mean(y_pred_classes[mask] == digit) * 100
                digit_count = np.sum(mask)
                results.append(f"{digit}: {digit_acc:.1f}% ({digit_count})")
                per_class_accuracies.append(digit_acc / 100.0)  # Convert to [0, 1] range
            else:
                # If class has no samples, skip it (shouldn't happen, but handle gracefully)
                per_class_accuracies.append(0.0)
        
        # Overall accuracy
        overall_acc = np.mean(y_pred_classes == self.y_val) * 100
        
        # Compute composite metric: Mean(VA) - λ * Variance(VA)
        if len(per_class_accuracies) == 10:
            mean_accuracy = np.mean(per_class_accuracies)
            variance = np.var(per_class_accuracies)  # This is (Σ(VA(i) - Mean(VA))²) / 10
            balanced_score = mean_accuracy - self.lambda_weight * variance
            
            # Add to logs so it can be monitored by EarlyStopping/ModelCheckpoint
            if logs is not None:
                logs['val_balanced_score'] = balanced_score
                logs['val_mean_class_accuracy'] = mean_accuracy
                logs['val_class_variance'] = variance
            
            print(f"  [Softmax10] Overall: {overall_acc:.1f}% | Per-digit: {' | '.join(results)}")
            print(f"  [Balanced] Mean: {mean_accuracy*100:.2f}% | Variance: {variance*10000:.2f} (×10⁻⁴) | Score: {balanced_score*100:.2f}%")
        else:
            print(f"  [Softmax10] Overall: {overall_acc:.1f}% | Per-digit: {' | '.join(results)}")


def load_dataset_from_npz(file_path, input_size, label_key, dataset_name):
    """
    Helper function to load a dataset from an npz file.
    
    Args:
        file_path: Path to the .npz file
        input_size: Expected image size (28 or 64)
        label_key: Key for labels in npz file (e.g. 'y')
        dataset_name: Name of dataset for error messages
    
    Returns:
        Tuple of (x_data, y_data)
        - x_data: (N, input_size, input_size, 1) float32 [0, 1]
        - y_data: (N,) int32
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data cannot be loaded or is invalid
    """
    if not file_path.exists():
        name = dataset_name or file_path.name
        raise FileNotFoundError(f"Dataset file not found: {file_path} (dataset: {name})")
    
    try:
        data = np.load(file_path)
        x_data = data['x'].astype(np.float32)
        y_data = data[label_key].astype(np.int32)
        
        # Check if data is valid
        if x_data is None or y_data is None:
            name = dataset_name or file_path.name
            raise ValueError(f"Invalid data in {file_path}: x_data or y_data is None (dataset: {name})")
        
        # Normalize if needed
        if x_data.max() > 1.0:
            x_data = x_data / 255.0
        
        # Reshape if needed
        if len(x_data.shape) == 3:
            x_data = x_data.reshape(-1, input_size, input_size, 1)
        
        return x_data, y_data
    except KeyError as e:
        name = dataset_name or file_path.name
        raise ValueError(f"Missing key '{e}' in {file_path} (dataset: {name})")
    except Exception as e:
        name = dataset_name or file_path.name
        raise ValueError(f"Could not load {name} from {file_path}: {e}")


def load_augmented_data(image_size, use_combined):
    """
    Load augmented data from the npz files.
    """
    if use_combined:
        dataset = "mnist_usps_ardis_customone"
    else:
        dataset = "mnist"
    augmented_train_path = DATA_DIR / "augmented" / f"{dataset}_train_augmented_{image_size}x{image_size}.npz"
    augmented_test_path = DATA_DIR / "augmented" / f"{dataset}_test_augmented_{image_size}x{image_size}.npz"
    x_train, y_train = load_dataset_from_npz(augmented_train_path, image_size, 'y', 'augmented train')
    x_test, y_test = load_dataset_from_npz(augmented_test_path, image_size, 'y', 'augmented test')
    return x_train, y_train, x_test, y_test


def calibrate_energy_scorer(classifier_model_path, 
    input_size, 
    use_combined, 
    model):
    """
    Load a pre-trained digit classifier and calibrate the energy scorer.
    """

    def load_augmented_dataX(image_size):
        return load_augmented_data(image_size, use_combined)

    print(f" For SoftMax11 model")
    print(f"Calibrating energy scorer for model: {classifier_model_path}...")
    print(f"input_size: {input_size}")
    print(f"use_combined: {use_combined}")
    
    calibrate_energy_scorer_helper(classifier_model_path, 
        load_augmented_dataX, 
        batch_size=BATCH_SIZE, 
        input_size=input_size,
        model=model)


def load_digit_classifier_and_energy_scorer(classifier_model_path, 
input_size):
    """
    Load a pre-trained digit classifier and energy scorer.
    
    Args:
        classifier_model_path: Path to the model file
        input_size: Image size for calibration if needed
    """
    model = load_digit_classifier(classifier_model_path)
    energy_scorer = _build_energy_model_from_path(model, 
    batch_size=BATCH_SIZE,
    classifier_model_path, percentile=99.5)

    return model, energy_scorer


def _new_model_with_banner(input_size, use_balanced_loss, lambda_weight, 
learning_rate, neurons_in_dense_layer, use_combined):
    print(f"Creating new digit classifier model for 10 classes (digits 0-9 only)...")
    print(f"  Input ({input_size}x{input_size} input.")
    if use_balanced_loss:
        print(f"  Loss: BalancedLoss with lambda weight: {lambda_weight}")
    else:
        print(f"  Loss: cross-entropy")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Dense layer neurons: {neurons_in_dense_layer}")
    print(f"  Use combined: {use_combined}")
    
    model = create_digit_classifier_model(
    input_size=input_size, 
    use_balanced_loss=use_balanced_loss, 
    lambda_weight=lambda_weight, 
    learning_rate=learning_rate, 
    neurons_in_dense_layer=neurons_in_dense_layer,
    use_combined=use_combined)

    return model


def train_digit_classifier( 
    num_epochs, input_size, initial_model_path,
    use_combined,
    use_balanced_loss, lambda_weight, learning_rate, neurons_in_dense_layer):
    """
    Load a pre-trained digit classifier or create/train a new one.
    
    Uses MNIST as base dataset with optional additional datasets and augmentation.
    
    Args:
        num_epochs: Number of training epochs (default: 20)
        input_size: Image size (28 or 64, default: 28)
        initial_model_path: Path to pre-trained model to use as starting point (optional)
        use_combined: Load all datasets (MNIST, USPS, ARDIS, CustomOne) or MNIST only
        use_balanced_loss: Use BalancedLoss in training (optimizes for balanced performance during backprop) (default: False)
        lambda_weight: Weight for variance penalty in BalancedLoss (default: 5.0)
        learning_rate: Learning rate for Adam optimizer (default: 0.0005)
        neurons_in_dense_layer: Number of neurons in final dense layer (default: 32)
        For classification uses softmax with 10 classes (digits 0-9 only)
        Always uses data augmentation and stratified batch sampling
        Load the pre-generated augmented data from the npz files
    
    Returns:
        Trained Keras model for digit classification
    """
    
    # Create timestamped run directory: run_YYYY_MM_DD_HH_MM_SS
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = DATA_DIR / "modelForDE" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model checkpoints will be saved to: {run_dir}")
    if initial_model_path is not None:
        if os.path.exists(initial_model_path):
            try:
                print(f"Loading initial model from: {initial_model_path}")
                print(f"  Will continue training from this checkpoint... ")

                # Provide custom objects in case model was saved with BalancedLoss
                model = keras.models.load_model(
                    initial_model_path,
                    custom_objects={'BalancedLoss': BalancedLoss}
                )
                print(f"  Initial model loaded successfully")
                # Recompile with new learning rate and loss function settings
                if use_balanced_loss:
                    loss_function = BalancedLoss(num_classes=10, 
                    var_penalty=lambda_weight,
                    digit_only_variance=True)
                else:
                    loss_function = 'sparse_categorical_crossentropy'
                    
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=loss_function,
                    metrics=['accuracy']
                )

                print(f"  Model recompiled with learning rate: {learning_rate}")
                if use_balanced_loss:
                    print(f"  Model using balanced loss with lambda weight: {lambda_weight}")
                    print(f"  Model using balanced loss digit only variance")

            except Exception as e:
                print(f"  Warning: Could not load initial model: {e}")
                print(f"  Creating new model instead...")
                model = _new_model_with_banner(input_size, use_balanced_loss, lambda_weight, 
                learning_rate, neurons_in_dense_layer, use_combined)
        else:
            print(f"Initial model path provided but file not found: {initial_model_path}")
            print(f"  Creating new model instead...")
            model = _new_model_with_banner(input_size, use_balanced_loss, lambda_weight, 
            learning_rate, neurons_in_dense_layer, use_combined)
    else:
        model = _new_model_with_banner(input_size, use_balanced_loss, lambda_weight, 
        learning_rate, neurons_in_dense_layer, use_combined)
        
    try:
        # =====================================================================
        # Load PRE-GENERATED augmented data (REQUIRED)
        # =====================================================================
        x_train, y_train, x_test, y_test = load_augmented_data(
            image_size=input_size,
            use_combined=use_combined,
        )

        if x_train is None:
            raise ValueError(f"Failed to load augmented data. Run augmentation scripts for size {input_size}")

        print("\n" + "="*60)
        print("FAST TRAINING MODE (pre-generated augmented data)")
        print("="*60)

        # Shuffle test data
        indices = np.random.permutation(len(x_test))
        x_test = x_test[indices]
        y_test = y_test[indices]
                
        print(f"Training samples: {len(x_train):,}")
        print(f"Test samples: {len(x_test):,}")
        print(f"Number of epochs: {num_epochs}")

        # =====================================================================
        # TRAINING SETUP
        # =====================================================================
        
        import time
        
        batch_size = BATCH_SIZE
        print(f"Epoch models will be saved as: {run_dir}/digit_classifier_epoch_XX.keras")
        
        # Early stopping settings
        patience = 5  # Stop if no improvement for 5 epochs
        min_delta = 0.0001  # Minimum change to qualify as improvement

        # =================================================================
        # FAST MODE: from_generator() with model.fit()
        # =================================================================
        
        steps_per_epoch = len(x_train) // batch_size
        val_steps = (len(x_test) + batch_size - 1) // batch_size
        print(f"Batch size: {batch_size}")
        print(f"Steps per epoch: {steps_per_epoch}")
        
        print("\nUsing STRATIFIED batch sampling (balanced classes per batch)")
        # Create the generator function (returns a callable that returns a generator)
        train_generator = create_stratified_batch_generator(
            x_train, y_train, batch_size=batch_size, 
            num_classes=10, non_digit_class=None
        )
        
        # Create tf.data.Dataset from generator (NO memory copy!)
        train_dataset = tf.data.Dataset.from_generator(
            train_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, input_size, input_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validation dataset - use from_tensor_slices for validation 
        #                      (references data, no copy)
        # Validation is small enough (~90K samples = ~1.4GB) that keeping in 
        # memory is acceptable
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(batch_size)

        softmax_callback = Softmax10DiagnosticsCallback(x_test, y_test, lambda_weight=lambda_weight)
        
        if use_balanced_loss:
            print(f"\nUsing VAL_LOSS (BalancedLoss) metric for monitoring:")
            print(f"  Loss = Mean(Class_Losses) + {lambda_weight} * Variance(Class_Losses)")
            print(f"  Minimizing: Mean class loss + {lambda_weight} * variance")
        else:
            print(f"\nUsing VAL_LOSS (cross-entropy) metric for monitoring (default)")

        #Callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=100, #patience,
                min_delta=min_delta,
                mode='min',
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / "digit_classifier_best.keras"),
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(run_dir / "digit_classifier_epoch_{epoch:02d}.keras"),
                save_freq='epoch',
                verbose=0
            ),
            softmax_callback,
        ] 
        
        # Train with model.fit()
        history = model.fit(
            train_dataset,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=val_steps,  # Explicitly set to prevent infinite loop with stratified
            callbacks=callbacks_list,
            verbose=1
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
        
        # Get predictions
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Per-class accuracy (softmax mode with 10 classes)
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


def main():
    """
    Standalone training function for the digit classifier.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train a digit classifier on MNIST dataset"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (required for --energy-calibrate and --create-logit-model)"
    )
    parser.add_argument(
        "--energy-calibrate",
        action="store_true",
        help="Calibrate energy model using the specified model"
    )
    parser.add_argument(
        "--create-logit-model",
        action="store_true",
        help="Create logit model from the specified model"
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="True means train model"
    )
    parser.add_argument(
        "--epoch-count",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        choices=[28, 64],
        help="Input image size (28 or 64, default: 28)"
    )
    parser.add_argument(
        "--initial-model",
        type=str,
        default=None,
        help="Path to pre-trained model to use as starting point for training"
    )
    #We always run in augmented mode and always use stratified batch sampling
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Include all datasets in training (MNIST, USPS, ARDIS, CustomOne). If not set, use MNIST only."
    )
    parser.add_argument(
        "--balanced-loss",
        action="store_true",
        help="Use BalancedLoss in training (optimizes for balanced performance during backpropagation)"
    )
    parser.add_argument(
        "--lambda-weight",
        type=float,
        default=5.0,
        help="Weight for variance penalty in BalancedLoss (default: 5.0)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate for Adam optimizer (default: 0.001)"
    )
    parser.add_argument(
        "--dense-layer",
        type=int,
        default=32,
        help="Number of neurons in final dense layer (default: 32)"
    )
    
    args = parser.parse_args()
    
    # Determine input size
    input_size = args.size
        
    # Check that exactly one option is selected
    options_selected = sum([args.train_model, args.energy_calibrate, args.create_logit_model])
    
    if options_selected > 1:
        print("Please specify only ONE option: --train-model, --energy-calibrate, or --create-logit-model")
        return 0

    if options_selected == 0:
        print("Please specify one of the following options:")
        print("  --train-model to train a new model")
        print("  --energy-calibrate to calibrate energy scorer for existing model")
        print("  --create-logit-model to create a logit model from an existing model")
        return 0

    if args.energy_calibrate:
        print(f"Starting energy model calibration...")
        print(f"Trained classifier model path: {args.model_path}")
        print(f"Input image size: {input_size}x{input_size}")
        calibrate_energy_scorer(args.model_path, 
        input_size=input_size, 
        use_combined=args.combined, 
        model=None)
        return 1

    if args.create_logit_model:
        print(f"Starting to create logit model from classifier model...")
        print(f"Trained classifier model path: {args.model_path}")
        create_logit_model(args.model_path)
        return 2   
    
    # Train the model (always uses softmax with 10 classes)
    if args.train_model:    
        print(f"Starting digit classifier training with 10 classes (digits 0-9 only)...")
        print(f"Input image size: {input_size}x{input_size}")
        print(f"Epochs: {args.epoch_count}")
        print(f"Initial model: {args.initial_model}")

        if args.combined:
            print(f"Datasets: MNIST, USPS, ARDIS, CustomOne")
        else:
            print(f"Datasets: MNIST")
        print()
        print(f"Data augmentation: ENABLED")
        print(f"Stratified batch sampling: ENABLED")
        print(f"Balanced loss (training): {'ENABLED' if args.balanced_loss else 'DISABLED'}")
        if args.balanced_loss:
            print(f"  Lambda weight: {args.lambda_weight}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Dense layer neurons: {args.dense_layer}")
        
        model = train_digit_classifier(
            num_epochs=args.epoch_count,
            input_size=input_size,
            initial_model_path=args.initial_model,
            use_combined=args.combined,
            use_balanced_loss=args.balanced_loss,
            lambda_weight=args.lambda_weight,
            learning_rate=args.learning_rate,
            neurons_in_dense_layer=args.dense_layer
        )
        print("\nTraining complete!")
        return 3

if __name__ == "__main__":
    main()
