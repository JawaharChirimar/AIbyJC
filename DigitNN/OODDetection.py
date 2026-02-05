#!/usr/bin/env python3
"""
OODDetection.py

Out-of-Distribution detection using signals from the existing CNN classifier.
No new model training required — extracts OOD scores from the trained model.

Two methods:
1. Energy Score — uses raw logits (pre-softmax), better than max softmax probability
2. Mahalanobis Distance — uses penultimate dense layer features (96-dim)

Usage:
    from OODDetection import EnergyScorer, MahalanobisScorer

    # --- Energy Score (zero setup) ---
    energy = EnergyScorer(model)
    score = energy.score(image)            # single image
    is_ood = energy.is_ood(image)          # True/False with default threshold
    scores = energy.score_batch(images)    # batch

    # --- Mahalanobis Distance (requires fitting on training data) ---
    maha = MahalanobisScorer(model)
    maha.fit(x_train, y_train)             # one-time: compute class means + covariance
    maha.save("maha_params.npz")           # save so you don't refit every time
    maha.load("maha_params.npz")           # load previously fitted params
    score = maha.score(image)              # single image
    scores = maha.score_batch(images)      # batch

    # --- Combined scoring ---
    combined = CombinedOODScorer(model, maha_params_path="maha_params.npz")
    result = combined.classify(image)
    # returns: {"class": 7, "confidence": 0.98, "energy": -12.3,
    #           "mahalanobis": 4.2, "is_ood": False}
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path


# =============================================================================
# HELPER: Build sub-models that extract intermediate outputs
# =============================================================================

def _build_logit_model(model):
    """
    Build a model that outputs raw logits (pre-softmax).
    
    Your architecture ends with:
        Dense(neurons, activation='elu')   ← penultimate
        BatchNormalization()
        Dropout(0.5)
        Dense(11, activation='softmax')  ← final
    
    We create a new model that's identical but outputs pre-softmax logits.
    """
    # Find the final Dense layer (the one with 11 units)
    final_dense = None
    final_dense_idx = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Dense) and layer.units == 11:
            final_dense = layer
            final_dense_idx = i
            break
    
    if final_dense is None:
        raise ValueError("Could not find final Dense(11) layer in model")
    
    if final_dense_idx == 0:
        raise ValueError("Final Dense layer is the first layer (unexpected architecture)")
    
    # Get the weights from the final Dense layer
    weights, biases = final_dense.get_weights()
    weights = tf.constant(weights, dtype=tf.float32)
    biases = tf.constant(biases, dtype=tf.float32)
    
    # Create a model that outputs everything up to (but not including) the final Dense layer
    # For Sequential models, we can use the layers directly
    if isinstance(model, keras.Sequential):
        # Get input shape - ensure model is built
        if model.input_shape is None:
            # Build model with dummy input
            dummy_shape = model.layers[0].input_shape[1:] if hasattr(model.layers[0], 'input_shape') else (28, 28, 1)
            dummy_input = np.zeros((1,) + dummy_shape, dtype=np.float32)
            _ = model(dummy_input)
        
        input_shape = model.input_shape[1:]  # Remove batch dimension
        intermediate_input = keras.Input(shape=input_shape)
        x = intermediate_input
        
        # Apply all layers up to (but not including) final_dense
        # We can reuse the existing layers - they're already built
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
    
    # Use Lambda layer to compute logits = W*x + b (no softmax)
    logits = keras.layers.Lambda(
        lambda x: tf.matmul(x, weights) + biases,
        name='logits'
    )(intermediate_output)
    
    # Create the logit model
    logit_model = keras.Model(inputs=intermediate_input, outputs=logits)
    
    return logit_model


def _build_feature_model(model):
    """
    Build a model that outputs the penultimate dense layer features.
    
    Extracts the output of Dense(96, activation='elu') — the 96-dim
    feature vector before BatchNorm/Dropout/final Dense.
    
    This is the representation your CNN learned for classification.
    """
    # Find the penultimate Dense layer (the one with neurons != 11)
    penultimate_dense = None
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Dense) and layer.units != 11:
            penultimate_dense = layer
            break
    
    if penultimate_dense is None:
        raise ValueError("Could not find penultimate Dense layer in model")
    
    feature_model = keras.Model(
        inputs=model.input,
        outputs=penultimate_dense.output
    )
    return feature_model


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
    
    def __init__(self, model, temperature=1.0):
        """
        Args:
            model: Your trained Keras classifier (with softmax output)
            temperature: Temperature for scaling logits (default 1.0, try 0.5-2.0)
        """
        self.model = model
        self.temperature = temperature
        self.logit_model = _build_logit_model(model)
        self._threshold = None  # Set via calibrate() or manually
    
    def _get_logits(self, images):
        """Get raw logits for a batch of images. Images should be preprocessed."""
        return self.logit_model.predict(images, verbose=0, batch_size=128)
    
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
    
    def set_threshold(self, threshold):
        """
        Manually set the energy threshold.
        
        Args:
            threshold: Energy threshold value. Images with energy > threshold are OOD.
        """
        self._threshold = threshold
    
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
    
    def calibrate(self, x_digits, percentile=[95]):
        """
        Calibrate the energy threshold using labeled data.
        
        Sets thresholds so that `percentile`% of real digits are accepted.
        
        Args:
            x_digits: Array of known digit images (shape: N, H, W, 1)
            percentile: What % of digits should be accepted. 
            An array of numbers where 1 < x < 99.99 (default [95])
        
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
    
    def load_calibration(self, model_path, percentile=99.9):
        """Load energy scorer calibration for model at model_path.
        threshold in the calibration file is a dictionary of percentiles and thresholds.
        Read from it the threshold for given percentile 
        and set it as the threshold for the energy scorer.
        If input percentile is not in the calibration file, use default value of 99.9."""

        file_path = self.calibration_file_name_from_model_path(model_path)
        print(f"Loading energy scorer calibration from: {file_path}")

        try:
            with open(file_path, 'r') as f:
                calibration = json.load(f)
            
            thresholds = calibration.get('thresholds', {})
            
            # If 99.9 is missing, raise exception (should never happen)
            if '99.9' not in thresholds:
                raise ValueError(f"load_calibration: Percentile 99.9 not found in calibration file {file_path}. This should never happen.")
            
            # Check if requested percentile exists, fall back to 99.9 if not
            percentile_str = str(percentile)
            if percentile_str not in thresholds:
                print(f"  Percentile {percentile} not found in calibration, using 99 instead")
                percentile_str = '99.9'
            
            self._threshold = thresholds[percentile_str]
            print(f"Energy scorer calibration loaded from: {file_path}")
            print(f"  Threshold: {self._threshold}")
            return self._threshold
        except Exception as e:
            raise ValueError(f"load_calibration: Failed to load calibration from {file_path}: {e}")
    
    def _prepare_input(self, image):
        """Reshape single image to (1, H, W, 1) batch."""
        img = np.array(image, dtype=np.float32)
        if img.ndim == 2:
            img = img.reshape(1, img.shape[0], img.shape[1], 1)
        elif img.ndim == 3:
            img = img.reshape(1, *img.shape)
        return img


# =============================================================================
# METHOD 2: Mahalanobis Distance
# =============================================================================

class MahalanobisScorer:
    """
    Mahalanobis distance-based OOD detection (Lee et al., NeurIPS 2018).
    
    Uses the penultimate dense layer features (96-dim) from your CNN.
    
    Setup (one-time):
        1. Run all training data through the CNN
        2. Extract 96-dim features from penultimate dense layer
        3. Compute per-class mean vectors (11 means, each 96-dim)
        4. Compute shared covariance matrix (96 x 96)
    
    At inference:
        1. Extract 96-dim feature for test image
        2. Compute Mahalanobis distance to each class centroid
        3. Return minimum distance (closest class)
    
    - In-distribution: SMALL distance (close to some class centroid)
    - Out-of-distribution: LARGE distance (far from all centroids)
    """
    
    def __init__(self, model):
        """
        Args:
            model: Your trained Keras classifier
        """
        self.model = model
        self.feature_model = _build_feature_model(model)
        self.feature_dim = self.feature_model.output_shape[-1]
        
        # Parameters (set by fit() or load())
        self.class_means = None      # shape: (num_classes, feature_dim)
        self.precision_matrix = None  # shape: (feature_dim, feature_dim) — inverse covariance
        self.num_classes = None
        self._threshold = None
    
    def fit(self, x_train, y_train, batch_size=256):
        """
        Compute class means and shared covariance from training data.
        
        This is the expensive step — run once, then save with save().
        
        Args:
            x_train: Training images, shape (N, H, W, 1), float32 [0, 1]
            y_train: Training labels, shape (N,), int
            batch_size: Batch size for feature extraction
        """
        print(f"Fitting Mahalanobis parameters on {len(x_train):,} samples...")
        
        # Extract features for all training data
        print(f"  Extracting {self.feature_dim}-dim features...")
        features = self.feature_model.predict(x_train, batch_size=batch_size, verbose=1)
        
        classes = np.unique(y_train)
        self.num_classes = len(classes)
        print(f"  Found {self.num_classes} classes: {classes}")
        
        # Compute per-class means
        self.class_means = np.zeros((self.num_classes, self.feature_dim))
        for i, c in enumerate(classes):
            mask = y_train == c
            self.class_means[i] = np.mean(features[mask], axis=0)
            print(f"    Class {c}: {np.sum(mask):,} samples, "
                  f"mean norm = {np.linalg.norm(self.class_means[i]):.4f}")
        
        # Compute shared covariance (tied covariance across all classes)
        print(f"  Computing shared covariance matrix ({self.feature_dim}x{self.feature_dim})...")
        centered = np.zeros_like(features)
        for i, c in enumerate(classes):
            mask = y_train == c
            centered[mask] = features[mask] - self.class_means[i]
        
        covariance = np.cov(centered.T)
        
        # Add small regularization for numerical stability
        covariance += np.eye(self.feature_dim) * 1e-6
        
        # Compute precision matrix (inverse covariance) — needed for Mahalanobis distance
        self.precision_matrix = np.linalg.inv(covariance)
        
        print(f"  Done. Covariance condition number: {np.linalg.cond(covariance):.2f}")
        print(f"  Mahalanobis parameters ready.")
    
    def save(self, path):
        """Save fitted parameters to .npz file."""
        if self.class_means is None:
            raise ValueError("No parameters to save. Call fit() first.")
        np.savez(path,
                 class_means=self.class_means,
                 precision_matrix=self.precision_matrix,
                 num_classes=self.num_classes)
        print(f"Mahalanobis parameters saved to: {path}")
    
    def load(self, path):
        """Load previously fitted parameters from .npz file."""
        data = np.load(path)
        self.class_means = data['class_means']
        self.precision_matrix = data['precision_matrix']
        self.num_classes = int(data['num_classes'])
        print(f"Mahalanobis parameters loaded from: {path}")
        print(f"  Classes: {self.num_classes}, Feature dim: {self.class_means.shape[1]}")
    
    def _compute_mahalanobis(self, features):
        """
        Compute minimum Mahalanobis distance to any class centroid.
        
        For each sample, computes:
            d_c = (f - μ_c)^T Σ^{-1} (f - μ_c)    for each class c
            score = min_c(d_c)
        
        Args:
            features: Array of shape (N, feature_dim)
        
        Returns:
            Array of shape (N,) — minimum Mahalanobis distance per sample
        """
        n_samples = features.shape[0]
        distances = np.zeros((n_samples, self.num_classes))
        
        for c in range(self.num_classes):
            diff = features - self.class_means[c]  # (N, D)
            # Mahalanobis: d = diff @ precision @ diff.T (per sample)
            left = diff @ self.precision_matrix  # (N, D)
            distances[:, c] = np.sum(left * diff, axis=1)  # (N,)
        
        # Return minimum distance (closest class)
        return np.min(distances, axis=1)
    
    def score(self, image):
        """
        Compute Mahalanobis distance for a single image.
        
        Returns:
            float: Min Mahalanobis distance. Larger = more likely OOD.
        """
        if self.class_means is None:
            raise ValueError("Parameters not fitted. Call fit() or load() first.")
        
        img = self._prepare_input(image)
        features = self.feature_model.predict(img, verbose=0)
        return float(self._compute_mahalanobis(features)[0])
    
    def score_batch(self, images, batch_size=256):
        """
        Compute Mahalanobis distances for a batch of images.
        
        Returns:
            numpy array of distances, shape (N,)
        """
        if self.class_means is None:
            raise ValueError("Parameters not fitted. Call fit() or load() first.")
        
        features = self.feature_model.predict(images, batch_size=batch_size, verbose=0)
        return self._compute_mahalanobis(features)
    
    def calibrate(self, x_digits, x_nondigits=None, percentile=95):
        """
        Calibrate threshold using labeled data.
        
        Args:
            x_digits: Known digit images
            x_nondigits: Optional known non-digit images
            percentile: What % of digits to accept (default 95)
        """
        digit_distances = self.score_batch(x_digits)
        self._threshold = np.percentile(digit_distances, percentile)
        
        print(f"Mahalanobis threshold at {percentile}th percentile: {self._threshold:.4f}")
        print(f"  Digit distances: mean={np.mean(digit_distances):.4f}, "
              f"std={np.std(digit_distances):.4f}")
        
        if x_nondigits is not None and len(x_nondigits) > 0:
            nondigit_distances = self.score_batch(x_nondigits)
            rejection_rate = np.mean(nondigit_distances > self._threshold) * 100
            print(f"  Non-digit distances: mean={np.mean(nondigit_distances):.4f}")
            print(f"  Non-digit rejection rate: {rejection_rate:.1f}%")
        
        return self._threshold
    
    def is_ood(self, image, threshold=None):
        """Check if image is OOD based on Mahalanobis distance."""
        thresh = threshold or self._threshold
        if thresh is None:
            raise ValueError("No threshold set. Call calibrate() first or pass threshold=")
        return self.score(image) > thresh
    
    def _prepare_input(self, image):
        """Reshape single image to (1, H, W, 1) batch."""
        img = np.array(image, dtype=np.float32)
        if img.ndim == 2:
            img = img.reshape(1, img.shape[0], img.shape[1], 1)
        elif img.ndim == 3:
            img = img.reshape(1, *img.shape)
        return img


# =============================================================================
# COMBINED: Use both methods together
# =============================================================================

class CombinedOODScorer:
    """
    Combines Energy Score + Mahalanobis Distance + existing softmax confidence.
    
    Three independent OOD signals from one model:
    1. Softmax confidence (what you already have)
    2. Energy score (better version of #1)
    3. Mahalanobis distance (feature-space distance)
    
    An image is flagged as OOD if ANY signal exceeds its threshold.
    """
    
    def __init__(self, model, maha_params_path=None, energy_temperature=1.0):
        """
        Args:
            model: Your trained Keras classifier
            maha_params_path: Path to saved Mahalanobis params (.npz).
                              If None, Mahalanobis scoring is disabled.
            energy_temperature: Temperature for energy score (default 1.0)
        """
        self.model = model
        self.energy_scorer = EnergyScorer(model, temperature=energy_temperature)
        
        self.maha_scorer = None
        if maha_params_path is not None:
            self.maha_scorer = MahalanobisScorer(model)
            self.maha_scorer.load(maha_params_path)
        
        # Default thresholds (set via calibrate())
        self.energy_threshold = None
        self.maha_threshold = None
        self.softmax_threshold = 0.5  # your existing confidence threshold
    
    def classify(self, image, input_size=28):
        """
        Full classification with OOD detection.
        
        Args:
            image: Preprocessed image (H, W) or (H, W, 1), float32 [0, 1]
            input_size: Expected image size
        
        Returns:
            dict with keys:
                - "class": predicted digit (0-9) or 10 (non-digit)
                - "confidence": softmax probability of predicted class
                - "energy": energy score (more negative = more likely digit)
                - "mahalanobis": Mahalanobis distance (smaller = more likely digit), or None
                - "is_ood_energy": True if energy says OOD
                - "is_ood_maha": True if Mahalanobis says OOD, or None
                - "is_ood_softmax": True if softmax confidence says OOD
                - "is_ood": True if ANY method flags OOD
        """
        img = np.array(image, dtype=np.float32)
        if img.ndim == 2:
            img_batch = img.reshape(1, img.shape[0], img.shape[1], 1)
        elif img.ndim == 3:
            img_batch = img.reshape(1, *img.shape)
        else:
            img_batch = img
        
        # Softmax prediction (existing behavior)
        predictions = self.model.predict(img_batch, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # Energy score
        energy = self.energy_scorer.score(img_batch)
        
        # Mahalanobis distance
        maha = None
        if self.maha_scorer is not None:
            maha = self.maha_scorer.score(img_batch)
        
        # OOD decisions
        is_ood_softmax = (predicted_class == 10) or (confidence < self.softmax_threshold)
        is_ood_energy = (self.energy_threshold is not None and energy > self.energy_threshold)
        is_ood_maha = None
        if maha is not None and self.maha_threshold is not None:
            is_ood_maha = maha > self.maha_threshold
        
        # Combined: OOD if ANY signal flags it
        is_ood = is_ood_softmax or is_ood_energy
        if is_ood_maha is not None:
            is_ood = is_ood or is_ood_maha
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "energy": energy,
            "mahalanobis": maha,
            "is_ood_softmax": is_ood_softmax,
            "is_ood_energy": is_ood_energy,
            "is_ood_maha": is_ood_maha,
            "is_ood": is_ood,
        }
    
    def calibrate(self, x_digits, x_nondigits=None, percentile=95):
        """
        Calibrate all thresholds at once.
        
        Args:
            x_digits: Known digit images (N, H, W, 1)
            x_nondigits: Known non-digit images (optional)
            percentile: Acceptance rate for digits
        """
        print("=" * 60)
        print("Calibrating OOD detection thresholds")
        print("=" * 60)
        
        print("\n--- Energy Score ---")
        self.energy_threshold = self.energy_scorer.calibrate(
            x_digits, percentile)
        
        if self.maha_scorer is not None:
            print("\n--- Mahalanobis Distance ---")
            self.maha_threshold = self.maha_scorer.calibrate(
                x_digits, x_nondigits, percentile)
        
        print("\n--- Thresholds ---")
        print(f"  Softmax confidence: < {self.softmax_threshold}")
        print(f"  Energy: > {self.energy_threshold:.4f}")
        if self.maha_threshold is not None:
            print(f"  Mahalanobis: > {self.maha_threshold:.4f}")


# =============================================================================
# STANDALONE: Quick test script
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Test OOD detection on trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .keras model")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of test images")
    parser.add_argument("--fit-maha", action="store_true", help="Fit Mahalanobis params on training data")
    parser.add_argument("--maha-params", type=str, default=None, help="Path to saved Mahalanobis .npz")
    parser.add_argument("--size", type=int, default=28, help="Input image size (28 or 64)")
    args = parser.parse_args()
    
    # Load model - determine which module to use based on model path
    model_path_str = str(args.model)
    if 'run_MNIST' in model_path_str:
        from DigitClassifierSoftMax10 import BalancedLoss
    elif 'run_EMNIST' in model_path_str:
        from DigitClassifierSoftMax11 import BalancedLoss
    else:
        # Default to SoftMax11 if pattern doesn't match
        from DigitClassifierSoftMax11 import BalancedLoss
    
    model = keras.models.load_model(
        args.model, custom_objects={'BalancedLoss': BalancedLoss})
    print(f"Model loaded: {args.model}")
    
    # Energy scorer — always available
    energy = EnergyScorer(model)
    print(f"Energy scorer ready")
    
    # Mahalanobis — fit or load
    if args.fit_maha:
        from DataManagement.PregenAugmentedData import load_augmented_data
        x_train, y_train, x_test, y_test = load_augmented_data(image_size=args.size)
        
        maha = MahalanobisScorer(model)
        maha.fit(x_train, y_train)
        
        save_path = os.path.splitext(args.model)[0] + "_maha_params.npz"
        maha.save(save_path)
        
        # Calibrate both
        digit_mask = y_test < 10
        nondigit_mask = y_test == 10
        
        print("\n--- Calibration ---")
        energy.calibrate(x_test[digit_mask])
        maha.calibrate(x_test[digit_mask], x_test[nondigit_mask])
    
    elif args.maha_params:
        maha = MahalanobisScorer(model)
        maha.load(args.maha_params)
    
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
            
            # Mahalanobis
            m = maha.score(img_batch) if (args.fit_maha or args.maha_params) else None
            
            m_str = f"  maha={m:.2f}" if m is not None else ""
            print(f"{fname:30s}  class={pred_class:2d}  conf={conf:.4f}  energy={e:.4f}{m_str}")
