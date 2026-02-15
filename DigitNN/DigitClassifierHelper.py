import os
import json
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from OODDetection import (LogitLayer, _is_logit_model, _build_logit_model, 
    _is_softmax_model, EnergyScorer)


class BalancedLoss(keras.losses.Loss):
    """
    Custom loss function that minimizes mean class loss + variance penalty.
    
    This encourages balanced performance across all classes by:
    1. Computing per-class mean losses
    2. Penalizing variance in class losses (higher variance = more imbalance)
    
    Formula: Loss = Mean(Class_Losses) + λ * Variance(Class_Losses)
    
    Where:
    - Mean(Class_Losses) = average cross-entropy loss across all classes
    - Variance(Class_Losses) = mean squared difference from the mean
    - λ = var_penalty parameter (controls strength of variance penalty)
    """
    def __init__(self, num_classes=11, var_penalty=0.5, 
    digit_only_variance=True,
    name='balanced_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
        self.num_classes = num_classes
        self.var_penalty = var_penalty
        self.digit_only_variance = digit_only_variance
        self.ce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )
    
    def call(self, y_true, y_pred):
        # Per-sample cross-entropy losses
        sample_losses = self.ce(y_true, y_pred)
        
        # Per-class mean loss
        class_losses = []
        class_counts = []
        for i in range(self.num_classes):
            mask = tf.cast(y_true == i, tf.float32)
            count = tf.reduce_sum(mask)
            class_counts.append(count)
            # Compute loss: sum(sample_losses * mask) / count
            # If count == 0, this will be 0/0 = NaN, which we'll filter out
            class_loss = tf.reduce_sum(sample_losses * mask) / (count + 1e-10)  # Small epsilon to avoid NaN
            class_losses.append(class_loss)
        
        class_losses = tf.stack(class_losses)
        class_counts = tf.stack(class_counts)
        
        # Filter to only include classes present in batch (count > 0)
        # NOTE: When use_stratified=True, both training and validation batches have all classes.
        # When use_stratified=False, batches can have missing classes (defensive handling).
        class_present = class_counts > 0
        valid_class_losses = tf.boolean_mask(class_losses, class_present)
        
        # If no classes present (shouldn't happen), return zero loss
        mean_loss = tf.cond(
            tf.size(valid_class_losses) > 0,
            lambda: tf.reduce_mean(valid_class_losses),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        
        if self.digit_only_variance and self.num_classes == 11:
            digit_losses = class_losses[:10]
            digit_counts = class_counts[:10]
            digit_present = digit_counts > 0
            valid_digit_losses = tf.boolean_mask(digit_losses, digit_present)
            variance = tf.cond(
                tf.size(valid_digit_losses) > 0,
                lambda: tf.reduce_mean(tf.square(valid_digit_losses - tf.reduce_mean(valid_digit_losses))),
                lambda: tf.constant(0.0, dtype=tf.float32))
        else:
            variance = tf.cond(
                tf.size(valid_class_losses) > 0,
                lambda: tf.reduce_mean(tf.square(valid_class_losses - mean_loss)),
                lambda: tf.constant(0.0, dtype=tf.float32))
        
        # Formula: Loss = Mean(Class_Losses) + λ * Variance(Class_Losses)
        penalty = self.var_penalty * variance
        
        return mean_loss + penalty

    def get_config(self):
        """Required for saving/loading models with custom loss."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'var_penalty': self.var_penalty,
            'digit_only_variance': self.digit_only_variance,
            'reduction': self.reduction
        })
        return config


def load_digit_classifier(classifier_model_path):
    """
    Load a pre-trained digit classifier.
    Can be used to load a softmax model or a logit model.
    
    Args:
        classifier_model_path: Path to the model file
    
    Returns:
        Keras model
    
    Raises:
        ValueError: If classifier_model_path is not provided or not found
        ValueError: If model cannot be loaded
    """
    if classifier_model_path is None or classifier_model_path == '' or not classifier_model_path:
        raise ValueError("load_digit_classifier:classifier_model_path must be provided when train_model=False")
    
    classifier_model_path = str(classifier_model_path)  # Convert Path to string if needed
    if not os.path.exists(classifier_model_path):
        raise ValueError(f"load_digit_classifier:Model file not found: {classifier_model_path}.")
    
    try:
        print(f"Loading digit classifier from: {classifier_model_path}")
        # Provide custom objects in case model was saved with BalancedLoss
        model = keras.models.load_model(
            classifier_model_path,
            custom_objects={'BalancedLoss': BalancedLoss, 'LogitLayer': LogitLayer}
        )
        
        print("Digit classifier loaded successfully")
    except Exception as e:
        raise ValueError(f"load_digit_classifier:Cannot load model from {classifier_model_path}: {e}. Set train_model=True to create a new model.")

    return model


def create_logit_model(classifier_model_path, model=None):
    """
    Load a pre-trained digit classifier and create a logit model from it.
    
    Args:
        classifier_model_path: Path to the model file
        model: Optional pre-loaded model (if None, will load from path)
    
    Raises:
        ValueError: If model cannot be loaded or logit model cannot be created
    """
    if model is None:
        model = load_digit_classifier(classifier_model_path)
    
    if model is None:
        raise ValueError(f"create_logit_model: Could not load model from {classifier_model_path}")

    if _is_logit_model(model):
        print("Model already outputs logits, no conversion needed.")
        return
    
    logit_model = _build_logit_model(model)
    if logit_model is None:
        raise ValueError(f"create_logit_model: Failed to build logit model from {classifier_model_path}")
    
    logit_model_path = classifier_model_path.replace('.keras', '_logit.keras')  
    try:
        logit_model.save(logit_model_path)
        print(f"Logit model saved to {logit_model_path}")
    except Exception as e:
        raise ValueError(f"create_logit_model: Failed to save logit model to {logit_model_path}: {e}")


def calibrate_energy_scorer_helper(
    classifier_model_path, 
    load_augmented_dataX, 
    batch_size, 
    input_size,
    model):
    """
    Load a pre-trained digit classifier and calibrate the energy scorer.
    """
    if model is None:
        model = load_digit_classifier(classifier_model_path)

    if model is None:
        raise ValueError(f"calibrate_energy_scorer_helper: Could not load model from {classifier_model_path}")
    
    if _is_softmax_model(model):
        raise ValueError(f"calibrate_energy_scorer_helper: Must be a logit model, not a softmax model. {classifier_model_path}")

    print("Loading test data for energy scorer calibration...")
    _, _, x_test, y_test = load_augmented_dataX(image_size=input_size)
        
    if x_test is None:
        raise ValueError(f"calibrate_energy_scorer_helper: Failed to load test data")
    else:
        # Filter to only digits (0-9), exclude non-digits (class 10)
        digit_mask = y_test < 10
        x_test_digits = x_test[digit_mask]

    energy_scorer = EnergyScorer(model, temperature=1.0, batch_size=batch_size)
    
    percentiles = [99.9, 99.5, 99, 95, 90]
    
    print(f"Calibrating energy scorer for percentiles: {percentiles}")
    thresholds = energy_scorer.calibrate(
        x_digits=x_test_digits, 
        percentile=percentiles)
    
    # Create dictionary mapping percentiles to thresholds
    thresholds_dict = {str(p): float(t) for p, t in zip(percentiles, thresholds)}
    
    # Create calibration data with thresholds and timestamp
    calibration_data = {
        'thresholds': thresholds_dict,
        'time': datetime.now().isoformat()
    }
    
    # Derive energy calibration file path from model path
    # e.g., /path/to/xy.keras -> /path/to/energy_xy_calibrate.json
    energy_file_path = EnergyScorer.calibration_file_name_from_model_path(classifier_model_path)
    
    # Write to JSON file (overwrite if exists)
    try:
        with open(energy_file_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Energy calibration saved to {energy_file_path}")
        return True
    except Exception as e:
        raise ValueError(f"calibrate_energy_scorer_helper: Failed to save calibration to {energy_file_path}: {e}")


def load_energy_caliberation_helper(calibration_file, energy_scorer):
    """
    Load the energy scorer calibration from a file.
    """
    if os.path.exists(calibration_file):
        print("Loading energy scorer calibration for model...")
        try:
            energy_scorer.load_calibration(calibration_file, percentile=99.5)
        except Exception as e:
            raise ValueError(f"load_energy_caliberation_helper: Could not load calibration: {e}")
    else:
        print("No calibration file found - calibration will need to be done separately")
        return -1
    
    return 0


def classify_digit(classifier_model, energy_scorer, digit_image, input_size=28):
    """
    Classify a single digit image using the CNN model.
    
    Args:
        classifier_model: Trained Keras model
        energy_scorer: EnergyScorer instance for OOD detection
        digit_image: Greyscale image (numpy array), will be resized to 
            input_size x input_size if needed
        input_size: Image size (28 or 64, default: 28)
    
    Returns:
        Tuple of (predicted_digit, confidence, score)
        - predicted_digit: int (0-9 for digits, 10 for "not a digit", 11 for OOD)
        - confidence: float (0.0-1.0) - probability of the predicted class
        - score: float - energy score
    """
    # Ensure image is the right shape and type
    if digit_image.shape != (input_size, input_size):
        # Resize using LANCZOS for quality
        digit_image = cv2.resize(
            digit_image, 
            (input_size, input_size), 
            interpolation=cv2.INTER_LANCZOS4
        )
    
    # Normalize pixel values to [0, 1]
    digit_normalized = digit_image.astype('float32') / 255.0
    
    # The input image should already be in MNIST format: white digits on black background
    # (ensured by BoundingBoxFromYolo.py preprocessing)
    
    # Reshape for model input: (1, input_size, input_size, 1)
    digit_input = digit_normalized.reshape(1, input_size, input_size, 1)
    
    # Predict
    if _is_logit_model(classifier_model):
        logits = classifier_model.predict(digit_input, verbose=0)
        predictions = tf.nn.softmax(logits).numpy()
    else:
        predictions = classifier_model.predict(digit_input, verbose=0)

    # Get predicted class (0-10)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

    score = energy_scorer.score(digit_input)
    print(f"Energy score: {score}")

    is_ood = energy_scorer.is_ood(digit_input)
    if is_ood:
        # Energy indicates OOD - even if classifier predicted a digit
        predicted_class = 11
        confidence = -1.0  # OOD detection, no confidence score
        print(f"  Energy-based OOD detection: Overriding to class 11 (non-digit)")
    else:
        print(f"  Energy-based OOD detection: Predicted class {predicted_class} is not OOD")
    
    # Return the predicted class (0-9 for digits, 10 for non-digit, 11 for OOD)
    return predicted_class, confidence, score