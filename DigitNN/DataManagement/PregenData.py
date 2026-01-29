#!/usr/bin/env python3
"""
PregenData.py

Pre-generates all upscaled datasets in one run.
Run this once before training to create cached versions.

Supports both 28x28 (default) and 64x64 image sizes.

Creates:
- data/MNIST/mnist_train_{size}x{size}.npz, mnist_test_{size}x{size}.npz
- data/EMNIST/emnist_digits_train_{size}x{size}.npz, emnist_digits_test_{size}x{size}.npz
- data/EMNIST/emnist_letters_train_{size}x{size}.npz
- data/usps/usps_train_{size}x{size}.npz, usps_test_{size}x{size}.npz
- data/ardis/ardis_train_{size}x{size}.npz, ardis_test_{size}x{size}.npz

Usage:
    python PregenData.py                    # Default: 28x28
    python PregenData.py --size 64          # 64x64
    python PregenData.py --size 28 --force  # Force regenerate
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Pre-generate upscaled datasets")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64],
                        help="Image size (28 or 64, default: 28)")
    parser.add_argument("--force", action="store_true", 
                        help="Force regenerate even if cached versions exist")
    args = parser.parse_args()
    
    force = args.force
    target_size = args.size
    
    print("="*60)
    print(f"Pre-generating {target_size}x{target_size} datasets")
    print("="*60)
    
    success = []
    failed = []
    
    # MNIST
    print("\n" + "-"*60)
    print("1. MNIST")
    print("-"*60)
    try:
        from DataManagement.GetMNIST import load_mnist_size
        result = load_mnist_size(target_size=target_size, force_regenerate=force)
        if result[0] is not None:
            success.append(f"MNIST: {result[0].shape}")
        else:
            failed.append("MNIST")
    except Exception as e:
        print(f"Error: {e}")
        failed.append("MNIST")
    
    # EMNIST Digits
    print("\n" + "-"*60)
    print("2. EMNIST Digits")
    print("-"*60)
    try:
        from DataManagement.GetEMNIST import load_emnist_size
        result = load_emnist_size(target_size=target_size, force_regenerate=force)
        if result[0] is not None:
            success.append(f"EMNIST Digits: {result[0].shape}")
        else:
            failed.append("EMNIST Digits")
    except Exception as e:
        print(f"Error: {e}")
        failed.append("EMNIST Digits")
    
    # EMNIST Letters (for non-digit generation)
    print("\n" + "-"*60)
    print("2b. EMNIST Letters (for non-digits)")
    print("-"*60)
    try:
        from DataManagement.GetEMNIST import load_emnist_letters_size
        result = load_emnist_letters_size(target_size=target_size, force=force)
        if result[0] is not None:
            success.append(f"EMNIST Letters: {result[0].shape}")
        else:
            failed.append("EMNIST Letters")
    except Exception as e:
        print(f"Error: {e}")
        failed.append("EMNIST Letters")
    
    # USPS
    print("\n" + "-"*60)
    print("3. USPS")
    print("-"*60)
    try:
        from DataManagement.GetUSPS import load_usps_size
        result = load_usps_size(target_size=target_size, force_regenerate=force)
        if result[0] is not None:
            success.append(f"USPS: {result[0].shape}")
        else:
            failed.append("USPS")
    except Exception as e:
        print(f"Error: {e}")
        failed.append("USPS")
    
    # ARDIS
    print("\n" + "-"*60)
    print("4. ARDIS")
    print("-"*60)
    try:
        from DataManagement.GetArdis import load_ardis_size
        result = load_ardis_size(target_size=target_size, force_regenerate=force)
        if result[0] is not None:
            success.append(f"ARDIS: {result[0].shape}")
        else:
            failed.append("ARDIS")
    except Exception as e:
        print(f"Error: {e}")
        failed.append("ARDIS")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if success:
        print("\n✓ Successfully generated:")
        for s in success:
            print(f"  - {s}")
    
    if failed:
        print("\n✗ Failed:")
        for f in failed:
            print(f"  - {f}")
        print("\nNote: Some datasets may need to be downloaded manually.")
        print("See individual Get*.py files for instructions.")
    
    print("\n" + "="*60)
    if not failed:
        print("All datasets ready!")
    else:
        print(f"Completed with {len(failed)} failure(s)")
    print("="*60)
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
