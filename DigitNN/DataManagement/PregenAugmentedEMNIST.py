#!/usr/bin/env python3
"""
PregenAugmentedEMNIST.py

Pre-generates augmented EMNIST training and test data.
EMNIST classes are already balanced, so we use a simple augmentation ratio.

Output:
- data/EMNIST/emnist_digits_train_augmented_{size}x{size}.npz
- data/EMNIST/emnist_digits_test_augmented_{size}x{size}.npz
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


EMNIST_DIR = DATA_DIR / "EMNIST"


def process_emnist_data(split='train', target_size=28, force=False):
    """Process EMNIST data with augmentation."""
    process_dataset_data(
        dataset_name="emnist_digits",
        dataset_dir=EMNIST_DIR,
        split=split,
        target_size=target_size,
        force=force,
        augment_ratio=AUGMENT_RATIO
    )


def main():
    parser = argparse.ArgumentParser(description="Pre-generate augmented EMNIST data")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Image size (28 or 64, default: 28)")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    parser.add_argument("--split", type=str, choices=['train', 'test', 'both'], default='both',
                        help="Which split to process (default: both)")
    
    args = parser.parse_args()
    
    if args.split in ['train', 'both']:
        process_emnist_data('train', args.size, args.force)
    
    if args.split in ['test', 'both']:
        process_emnist_data('test', args.size, args.force)
    
    print(f"\n{'='*70}")
    print("All done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
