#!/usr/bin/env python3
"""
GenerateAllAugmentedData.py

Orchestrates the complete augmented data generation pipeline.
Runs all augmentation steps in the correct order and automatically
calculates train/test counts for non-digit generation.

Usage:
    python3 DataManagement/GenerateAllAugmentedData.py --size 64
    python3 DataManagement/GenerateAllAugmentedData.py --size 64 --force
    python3 DataManagement/GenerateAllAugmentedData.py --size 28 --augment-ratio 30
    python3 DataManagement/GenerateAllAugmentedData.py --size 64 --skip-emnist --force
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import generation functions from individual scripts
from DataManagement.PregenAugmentedDataMNIST import create_augmented_data
from DataManagement.PregenGoogleFonts import generate_digit_images
from DataManagement.PregenNonDigits import generate_and_save


def main():
    parser = argparse.ArgumentParser(
        description="Generate all augmented datasets in the correct order"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        choices=[28, 64],
        help="Image size (28 or 64, default: 64)"
    )
    parser.add_argument(
        "--augment-ratio",
        type=int,
        default=50,
        help="Augmentation ratio percentage (default: 50)"
    )
    parser.add_argument(
        "--skip-emnist",
        action="store_true",
        help="Skip EMNIST augmentation (if already generated)"
    )
    parser.add_argument(
        "--skip-ardis",
        action="store_true",
        help="Skip ARDIS augmentation (if already generated)"
    )
    parser.add_argument(
        "--skip-usps",
        action="store_true",
        help="Skip USPS augmentation (if already generated)"
    )
    parser.add_argument(
        "--skip-fonts",
        action="store_true",
        help="Skip Google Fonts augmentation (if already generated)"
    )
    parser.add_argument(
        "--skip-nondigits",
        action="store_true",
        help="Skip non-digits augmentation"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cached files exist"
    )

    args = parser.parse_args()

    size = args.size
    augment_ratio = args.augment_ratio / 100.0  # Convert to decimal
    force = args.force

    print("\n" + "="*70)
    print("AUGMENTED DATA GENERATION PIPELINE")
    print("="*70)
    print(f"Image size: {size}x{size}")
    print(f"Augmentation ratio: {int(augment_ratio*100)}%")
    print(f"Force regeneration: {force}")
    print("="*70)

    success_steps = []
    failed_steps = []

    # Track actual counts from each dataset
    digit_counts = {
        "emnist": {"train": 0, "test": 0},
        "ardis": {"train": 0, "test": 0},
        "usps": {"train": 0, "test": 0},
        "fonts": {"train": 0, "test": 0}
    }

    # -------------------------------------------------------------------------
    # Step 1: Generate augmented EMNIST
    # -------------------------------------------------------------------------
    if not args.skip_emnist:
        print("\n" + "="*70)
        print("STEP 1: Generate augmented EMNIST")
        print("="*70)
        try:
            train_count, test_count = create_augmented_data(
                augment_ratio=augment_ratio,
                image_size=size,
                force=force,
                datasets='emnist'
            )
            digit_counts["emnist"]["train"] = train_count
            digit_counts["emnist"]["test"] = test_count
            success_steps.append("EMNIST")
            print(f"\n✓ EMNIST generation complete")
            print(f"  Train: {train_count:,} samples")
            print(f"  Test: {test_count:,} samples")
        except Exception as e:
            print(f"\n✗ EMNIST generation failed: {e}")
            failed_steps.append("EMNIST")
    else:
        print("\n⏭️  Skipping EMNIST (use --force to regenerate)")

    # -------------------------------------------------------------------------
    # Step 2: Generate augmented ARDIS
    # -------------------------------------------------------------------------
    if not args.skip_ardis:
        print("\n" + "="*70)
        print("STEP 2: Generate augmented ARDIS")
        print("="*70)
        try:
            train_count, test_count = create_augmented_data(
                augment_ratio=augment_ratio,
                image_size=size,
                force=force,
                datasets='ardis'
            )
            digit_counts["ardis"]["train"] = train_count
            digit_counts["ardis"]["test"] = test_count
            success_steps.append("ARDIS")
            print(f"\n✓ ARDIS generation complete")
            print(f"  Train: {train_count:,} samples")
            print(f"  Test: {test_count:,} samples")
        except Exception as e:
            print(f"\n✗ ARDIS generation failed: {e}")
            failed_steps.append("ARDIS")
    else:
        print("\n⏭️  Skipping ARDIS (use --force to regenerate)")

    # -------------------------------------------------------------------------
    # Step 3: Generate augmented USPS
    # -------------------------------------------------------------------------
    if not args.skip_usps:
        print("\n" + "="*70)
        print("STEP 3: Generate augmented USPS")
        print("="*70)
        try:
            train_count, test_count = create_augmented_data(
                augment_ratio=augment_ratio,
                image_size=size,
                force=force,
                datasets='usps'
            )
            digit_counts["usps"]["train"] = train_count
            digit_counts["usps"]["test"] = test_count
            success_steps.append("USPS")
            print(f"\n✓ USPS generation complete")
            print(f"  Train: {train_count:,} samples")
            print(f"  Test: {test_count:,} samples")
        except Exception as e:
            print(f"\n✗ USPS generation failed: {e}")
            failed_steps.append("USPS")
    else:
        print("\n⏭️  Skipping USPS (use --force to regenerate)")

    # -------------------------------------------------------------------------
    # Step 4: Generate augmented Google Fonts
    # -------------------------------------------------------------------------
    if not args.skip_fonts:
        print("\n" + "="*70)
        print("STEP 4: Generate augmented Google Fonts")
        print("="*70)
        try:
            # Get API key from environment
            api_key = os.environ.get("GOOGLE_FONTS_API_KEY")
            if not api_key:
                print("\n✗ Google Fonts API key not found!")
                print("Set GOOGLE_FONTS_API_KEY environment variable")
                print("or run PregenGoogleFonts.py separately with --api-key")
                failed_steps.append("Google Fonts")
            else:
                (x_train, y_train), (x_test, y_test) = generate_digit_images(
                    api_key=api_key,
                    output_dir="./data/font_digits",
                    target_size=size
                )
                train_count = len(x_train)
                test_count = len(x_test)
                digit_counts["fonts"]["train"] = train_count
                digit_counts["fonts"]["test"] = test_count
                success_steps.append("Google Fonts")
                print(f"\n✓ Google Fonts generation complete")
                print(f"  Train: {train_count:,} samples")
                print(f"  Test: {test_count:,} samples")
        except Exception as e:
            print(f"\n✗ Google Fonts generation failed: {e}")
            failed_steps.append("Google Fonts")
    else:
        print("\n⏭️  Skipping Google Fonts (use --force to regenerate)")

    # -------------------------------------------------------------------------
    # Calculate totals for non-digit generation
    # -------------------------------------------------------------------------
    digit_train_total = int(sum(digit_counts[k]["train"] for k in digit_counts)/10)
    digit_test_total = int(sum(digit_counts[k]["test"] for k in digit_counts)/10)

    print("\n" + "="*70)
    print("DIGIT DATA TOTALS (for non-digit matching)")
    print("="*70)
    print(f"Total train samples: {digit_train_total:,}")
    print(f"Total test samples: {digit_test_total:,}")
    print("\nBreakdown:")
    for dataset in ["emnist", "ardis", "usps", "fonts"]:
        train = digit_counts[dataset]["train"]
        test = digit_counts[dataset]["test"]
        if train > 0 or test > 0:
            print(f"  {dataset.upper():8s}: {train:,} train, {test:,} test")
    print("="*70)

    # -------------------------------------------------------------------------
    # Step 5: Generate augmented Non-digits
    # -------------------------------------------------------------------------
    if not args.skip_nondigits:
        if digit_train_total == 0 or digit_test_total == 0:
            print("\n✗ Cannot generate non-digits: No digit data was generated")
            print("Run steps 1-4 first to get digit counts")
            failed_steps.append("Non-digits")
        else:
            print("\n" + "="*70)
            print("STEP 5: Generate augmented Non-digits")
            print("="*70)
            try:
                generate_and_save(
                    force=force,
                    image_size=size,
                    train_count=digit_train_total,
                    test_count=digit_test_total
                )
                success_steps.append("Non-digits")
                print(f"\n✓ Non-digits generation complete")
                print(f"  Train: {digit_train_total:,} samples")
                print(f"  Test: {digit_test_total:,} samples")
            except Exception as e:
                print(f"\n✗ Non-digits generation failed: {e}")
                failed_steps.append("Non-digits")
    else:
        print("\n⏭️  Skipping Non-digits")

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)

    if success_steps:
        print(f"\n✓ Successfully generated ({len(success_steps)}):")
        for step in success_steps:
            print(f"  - {step}")

    if failed_steps:
        print(f"\n✗ Failed ({len(failed_steps)}):")
        for step in failed_steps:
            print(f"  - {step}")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Verify the generated files in data/ directories:")
    print(f"   - data/EMNIST/emnist_digits_train_augmented_{size}x{size}.npz")
    print(f"   - data/ardis/ardis_train_augmented_{size}x{size}.npz")
    print(f"   - data/usps/usps_train_augmented_{size}x{size}.npz")
    print(f"   - data/font_digits/font_digits_train_augmented_{size}x{size}.npz")
    print(f"   - data/non_digits_pregen/non_digits_train_augmented_{size}x{size}.npz")
    print("   (and corresponding test files)")
    print("\n2. Use PregenAugmentedData.py to load and combine all datasets")
    print("\n3. Train your model with the augmented data")
    print("\n" + "="*70)

    return 0 if not failed_steps else 1


if __name__ == "__main__":
    sys.exit(main())
