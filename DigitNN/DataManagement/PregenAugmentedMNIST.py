#!/usr/bin/env python3
"""
PregenAugmentedMNIST.py

Pre-generates augmented MNIST training and test data.
MNIST classes are already balanced, so we use a simple augmentation ratio.

Output:
- data/MNIST/mnist_train_augmented_{size}x{size}.npz
- data/MNIST/mnist_test_augmented_{size}x{size}.npz
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import common functions from base module
from DataManagement.PregenAugmentedBase import (
    process_dataset_data,
    DATA_DIR,
    AUGMENT_RATIO
)


MNIST_DIR = DATA_DIR / "MNIST"


def process_mnist_data(split='train', target_size=28, force=False):
    """Process MNIST data with augmentation."""
    process_dataset_data(
        dataset_name="mnist",
        dataset_dir=MNIST_DIR,
        split=split,
        target_size=target_size,
        force=force,
        augment_ratio=AUGMENT_RATIO
    )


def main():
    parser = argparse.ArgumentParser(description="Pre-generate augmented MNIST data")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Image size (28 or 64, default: 28)")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    parser.add_argument("--split", type=str, choices=['train', 'test', 'both'], default='both',
                        help="Which split to process (default: both)")
    
    args = parser.parse_args()
    
    if args.split in ['train', 'both']:
        process_mnist_data('train', args.size, args.force)
    
    if args.split in ['test', 'both']:
        process_mnist_data('test', args.size, args.force)
    
    print(f"\n{'='*70}")
    print("All done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
