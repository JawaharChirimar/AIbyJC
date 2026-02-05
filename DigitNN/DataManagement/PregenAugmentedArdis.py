#!/usr/bin/env python3
"""
PregenAugmentedArdis.py

Pre-generates augmented ARDIS training and test data.
ARDIS classes are already balanced, so we use a simple augmentation ratio.

Output:
- data/ardis/ardis_train_augmented_{size}x{size}.npz
- data/ardis/ardis_test_augmented_{size}x{size}.npz
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import common functions from base module
from DataManagement.PregenAugmentedBase import (
    process_dataset_data,
    DATA_DIR
)

ARDIS_DIR = DATA_DIR / "ardis"

# Augmentation ratio (10% of each class, matching PregenAugmentedData)
AUGMENT_RATIO = 0.10

def process_ardis_data(split='train', target_size=28, force=False):
    """Process ARDIS data with augmentation."""
    process_dataset_data(
        dataset_name="ardis",
        dataset_dir=ARDIS_DIR,
        split=split,
        target_size=target_size,
        force=force,
        augment_ratio=AUGMENT_RATIO
    )


def main():
    parser = argparse.ArgumentParser(description="Pre-generate augmented ARDIS data")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Image size (28 or 64, default: 28)")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    parser.add_argument("--split", type=str, choices=['train', 'test', 'both'], default='both',
                        help="Which split to process (default: both)")
    
    args = parser.parse_args()
    
    if args.split in ['train', 'both']:
        process_ardis_data('train', args.size, args.force)
    
    if args.split in ['test', 'both']:
        process_ardis_data('test', args.size, args.force)
    
    print(f"\n{'='*70}")
    print("All done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
