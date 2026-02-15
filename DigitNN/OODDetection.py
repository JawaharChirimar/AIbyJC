#!/usr/bin/env python3
"""
OODDetection.py

Out-of-Distribution detection using signals from the existing CNN classifier.
No new model training required — extracts OOD scores from the trained model.

Energy Score — uses raw logits (pre-softmax), better than max softmax probability

Usage:
    from OODDetection import EnergyScorer

    # --- Energy Score (zero setup) ---
    energy = EnergyScorer(model)
    score = energy.score(image)            # single image
    is_ood = energy.is_ood(image)          # True/False with default threshold
    scores = energy.score_batch(images)    # batch
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path


# =============================================================================
# Custom Layer for logit extraction (serialization-safe, no Lambda issues)
# =============================================================================

@keras.utils.register_keras_serializable(package="OODDetection")
class LogitLayer(keras.layers.Layer):
    """Custom layer that computes logits = W*x + b (no activation)."""
    
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="zeros",
            trainable=False,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=False,
            name="bias"
        )
        super().build(input_shape)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# =============================================================================
# HELPER: Build sub-models that extract intermediate outputs
# =============================================================================

def _is_softmax_model(model):
    """
    Check if the model's final layer has softmax activation.
    
    Returns:
        True if model outputs softmax probabilities (old models)
        False if model outputs logits (new models)
    """
    last_layer = model.layers[-1]
    if hasattr(last_layer, 'activation'):
        activation_name = last_layer.activation.__name__
        return activation_name == 'softmax'
    return False  # Assume logits if no activation info

def _is_logit_model(model):
    return not _is_softmax_model(model)

def _build_logit_model(model):
    """
    Build a model that outputs raw logits (pre-softmax) from a softmax model.
    
    Only needed for OLD models that have softmax baked into the final layer.
    For NEW models (logit output), use model directly.
    
    Architecture:
        ... → Dense(neurons, activation='elu') → BatchNorm → Dropout → Dense(N, activation='softmax')
    
    This function creates a new model that outputs pre-softmax logits by:
        1. Finding the final Dense layer (output layer)
        2. Getting its weights (W, b)
        3. Building a model that computes W*x + b without softmax
    """
    # Find the final Dense layer (last one in the model - the output layer)
    final_dense = None
    final_dense_idx = None
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if isinstance(layer, keras.layers.Dense):
            final_dense = layer
            final_dense_idx = i
            break
    
    if final_dense is None:
        raise ValueError("Could not find final Dense layer in model")
    
    if final_dense_idx == 0:
        raise ValueError("Final Dense layer is the first layer (unexpected architecture)")
    
    # Get the weights from the final Dense layer
    weights, biases = final_dense.get_weights()
    weights = tf.constant(weights, dtype=tf.float32)
    biases = tf.constant(biases, dtype=tf.float32)
    
    # Create a model that outputs everything up to (but not including) the final Dense layer
    if isinstance(model, keras.Sequential):
        # Get input shape - ensure model is built
        if model.input_shape is None:
            dummy_shape = model.layers[0].input_shape[1:] if hasattr(model.layers[0], 'input_shape') else (28, 28, 1)
            dummy_input = np.zeros((1,) + dummy_shape, dtype=np.float32)
            _ = model(dummy_input)
        
        input_shape = model.input_shape[1:]  # Remove batch dimension
        intermediate_input = keras.Input(shape=input_shape)
        x = intermediate_input
        
        # Apply all layers up to (but not including) final_dense
        for layer in model.layers[:final_dense_idx]:
            x = layer(x)
        
        intermediate_output = x
    else:
        # For Functional API: create model that outputs the layer before final_dense
        intermediate_model = keras.Model(
            inputs=model.input,
            outputs=model.layers[final_dense_idx - 1].output
        )
        intermediate_output = intermediate_model.output
        intermediate_input = model.input
    
    # Use custom LogitLayer instead of Lambda (serialization-safe)
    num_classes = final_dense.units
    logit_layer = LogitLayer(units=num_classes, name='logits')
    logits = logit_layer(intermediate_output)
    
    # Create the logit model
    logit_model = keras.Model(inputs=intermediate_input, outputs=logits)
    
    # Set the weights from the original final Dense layer
    logit_layer.set_weights([weights.numpy(), biases.numpy()])
    
    return logit_model

def _build_energy_model_from_path(model, 
batch_size, classifier_model_path, percentile):
    _energy_scorer = EnergyScorer(model, batch_size)
        
    if _is_softmax_model(model):
        raise ValueError(f"_build_energy_model_from_path: Model is softmax. Use LogitModel instead. {classifier_model_path}")

    calibration_file = _energy_scorer.calibration_file_name_from_model_path(classifier_model_path)
    if os.path.exists(calibration_file):
        print("Loading energy scorer calibration for model...")
        _energy_scorer.load_calibration(calibration_file, percentile=percentile)
    else:
        print(f"OODDetection: No calibration file found for model {classifier_model_path} - calibration will need to be done separately")
        raise ValueError(f"OODDetection: No calibration file found for model {classifier_model_path} - calibration will need to be done separately")
    
    return _energy_scorer


# =============================================================================
# METHOD 1: Energy Score
# =============================================================================

class EnergyScorer:
    """
    Energy-based OOD detection (Liu et al., NeurIPS 2020).
    
    Energy(x) = -log( Σ exp(logit_i) )
    
    - In-distribution (digits): LOW energy (large negative, e.g., -15)
    - Out-of-distribution:      HIGH energy (closer to 0, e.g., -3)
    
    Better than max softmax probability because energy is theoretically
    aligned with input density. Softmax normalizes away magnitude info.
    
    Temperature T can sharpen the separation (default T=1).
    """
    
    def __init__(self, model,  batch_size, temperature=1.0):
        """
        Args:
            model: Your trained Keras classifier (softmax or logit output)
            temperature: Temperature for scaling logits (default 1.0, try 0.5-2.0)
            batch_size: Batch size for predictions (default 128)
        """
        self.model = model
        self.temperature = temperature
        self.batch_size = batch_size
        self._threshold = None  # Set via calibrate() or manually
        
        # Detect model type
        #self._is_softmax = _is_softmax_model(model)
        if _is_softmax_model(model):
            print("  EnergyScorer does not support softmax models. Use LogitModel instead.")
            #self._logit_model = _build_logit_model(model)
            raise ValueError(f"EnergyScorer does not support softmax models. Use LogitModel instead.")
        else:
            self._logit_model = model
    
    def _get_logits(self, images):
        """Get raw logits for a batch of images. Images should be preprocessed."""
        return self._logit_model.predict(images, verbose=0, batch_size=self.batch_size)
    
    def _compute_energy(self, logits):
        """
        Compute energy score from logits.
        
        E(x) = -T * log( Σ exp(logit_i / T) )
        
        Lower energy = more likely in-distribution.
        """
        T = self.temperature
        scaled = logits / T
        max_scaled = np.max(scaled, axis=1, keepdims=True)
        lse = max_scaled.squeeze(axis=1) + np.log(
            np.sum(np.exp(scaled - max_scaled), axis=1))
        return -T * lse
    
    def score(self, image):
        """
        Compute energy score for a single image.
        
        Args:
            image: Preprocessed image array, shape (H, W) or (H, W, 1) or (1, H, W, 1)
                   Values should be float32 in [0, 1]
        
        Returns:
            float: Energy score. More negative = more likely a digit.
        """
        img = self._prepare_input(image)
        logits = self._get_logits(img)
        return float(self._compute_energy(logits)[0])
    
    def score_batch(self, images):
        """
        Compute energy scores for a batch of images.
        
        Args:
            images: Array of shape (N, H, W, 1), float32 in [0, 1]
        
        Returns:
            numpy array of energy scores, shape (N,)
        """
        logits = self._get_logits(images)
        return self._compute_energy(logits)
    
    
    def is_ood(self, image, threshold=None):
        """
        Determine if an image is out-of-distribution.
        
        Args:
            image: Single preprocessed image
            threshold: Energy threshold. If None, uses calibrated threshold.
                       Images with energy > threshold are OOD.
        
        Returns:
            bool: True if image is OOD (not a digit)
        """
        thresh = threshold or self._threshold
        if thresh is None:
            raise ValueError("No threshold set. Call calibrate() first or pass threshold=")
        return self.score(image) > thresh
    
    def calibrate(self, x_digits, percentile):
        """
        Calibrate the energy threshold using labeled data.
        
        Sets thresholds so that `percentile`% of real digits are accepted.
        
        Args:
            x_digits: Array of known digit images (shape: N, H, W, 1)
            percentile: What % of digits should be accepted. 
            An array of numbers where 1 < x < 99.99
        
        Returns:
            returns array of thresholds corresponding to each percentile
        """
        percentile = np.array(percentile)
        
        # Validate percentile range: 1 < x < 99.99
        if np.any(percentile <= 1) or np.any(percentile >= 99.99):
            raise ValueError("All percentiles must satisfy: 1 < percentile < 99.99")
        
        print(f"  Computing energy scores on {len(x_digits):,} samples (this may take a minute)...")
        digit_energies = self.score_batch(x_digits)
        
        # Threshold = energy value at the given percentile(s) of digit distribution
        # (digits have low/negative energy, so we want the high end of digit energies)
        thresholds = np.percentile(digit_energies, percentile)
        
        # Set the first threshold as the default (for backward compatibility)
        self._threshold = thresholds[0]
        
        print(f"Energy thresholds calibrated for {len(percentile)} percentiles:")
        for p, t in zip(percentile, thresholds):
            print(f"  {p:6.2f}th percentile: {t:.4f}")
        
        print(f"  Digit energy:  mean={np.mean(digit_energies):.4f}, "
              f"std={np.std(digit_energies):.4f}, "
              f"range=[{np.min(digit_energies):.4f}, {np.max(digit_energies):.4f}]")
        
        return thresholds

    @staticmethod
    def calibration_file_name_from_model_path(model_path):
        """Get calibration file name from model path."""
        model_path = Path(model_path)
        base_name = model_path.stem  # Gets 'xy' from 'xy.keras'
        energy_file_path = model_path.parent / f"energy_{base_name}_calibrate.json"
        return str(energy_file_path)
    
    def load_calibration(self, calibration_file, percentile):
        """Load energy scorer calibration for model at model_path.
        threshold in the calibration file is a dictionary of percentiles and thresholds.
        Read from it the threshold for given percentile 
        and set it as the threshold for the energy scorer.
        If input percentile is not in the calibration file, use 99.9 as default."""

        print(f"Loading energy scorer calibration from: {calibration_file}")

        try:
            with open(calibration_file, 'r') as f:
                calibration = json.load(f)
            
            thresholds = calibration.get('thresholds', {})
            
            # If 99.9 is missing, raise exception (should never happen)
            if '99.9' not in thresholds:
                raise ValueError(f"load_calibration: Percentile 99.9 not found in calibration file {calibration_file}. This should never happen.")
            
            # Check if requested percentile exists, fall back to 99.9 if not
            percentile_str = str(percentile)
            if percentile_str not in thresholds:
                print(f"  Percentile {percentile} not found in calibration, using 99.9 instead")
                percentile_str = '99.9'
            
            self._threshold = thresholds[percentile_str]
            print(f"Energy scorer calibration loaded from: {calibration_file}")
            print(f"  Threshold: {self._threshold}")
            return self._threshold
        except Exception as e:
            raise ValueError(f"load_calibration: Failed to load calibration from {calibration_file}: {e}")
    
    def _prepare_input(self, image):
        """Reshape single image to (1, H, W, 1) batch."""
        img = np.array(image, dtype=np.float32)
        if img.ndim == 2:
            img = img.reshape(1, img.shape[0], img.shape[1], 1)
        elif img.ndim == 3:
            img = img.reshape(1, *img.shape)
        return img



# =============================================================================
# STANDALONE: Quick test script
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OOD detection on trained model")
    parser.add_argument("--model", type=str, required=True, 
        help="Path to trained .keras model")
    parser.add_argument("--image-dir", type=str, default=None, 
    help="Directory of test images")
    parser.add_argument("--size", type=int, default=64, 
    help="Input image size (28 or 64)")
    args = parser.parse_args()
        
    # Load model - determine which module to use based on model path
    model_path_str = str(args.model)
    if 'run_MNIST' in model_path_str:
        from DigitClassifierSoftMax10 import BalancedLoss
    elif 'run_EMNIST' in model_path_str:
        from DigitClassifierSoftMax11 import BalancedLoss
    else:
        raise ValueError(f"OODDetection: Invalid model path: {model_path_str}. Must be one of: run_MNIST*, run_EMNIST*")

    model = keras.models.load_model(
        args.model, custom_objects={'BalancedLoss': BalancedLoss})
    print(f"Model loaded: {args.model}")
    
    # Energy scorer — always available
    energy = EnergyScorer(model, batch_size)
    print(f"Energy scorer ready")
        
    # Test on images if provided
    if args.image_dir:
        import cv2
        image_dir = args.image_dir
        for fname in sorted(os.listdir(image_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            img_path = os.path.join(image_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img_resized = cv2.resize(img, (args.size, args.size),
                                     interpolation=cv2.INTER_LANCZOS4)
            img_norm = img_resized.astype('float32') / 255.0
            img_batch = img_norm.reshape(1, args.size, args.size, 1)
            
            # Softmax prediction
            preds = model.predict(img_batch, verbose=0)
            pred_class = int(np.argmax(preds[0]))
            conf = float(preds[0][pred_class])
            
            # Energy
            e = energy.score(img_batch)
            
            print(f"{fname:30s}  class={pred_class:2d}  conf={conf:.4f}  energy={e:.4f}")
