#!/usr/bin/env python3
"""
Visualize filters (weights) from the first Conv2D layer of a trained model.
"""

import sys
import numpy as np
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt

def visualize_first_conv_filters(model_path, output_path=None, num_filters=None):
    """
    Visualize the filters from the first Conv2D layer of a trained model.
    
    Args:
        model_path: Path to the trained .keras model
        output_path: Path to save the visualization (optional)
        num_filters: Number of filters to display (None = all)
    """
    # Load the model
    print(f"Loading model from: {model_path}")
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find the first Conv2D layer
    first_conv = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            first_conv = layer
            break
    
    if first_conv is None:
        print("Error: No Conv2D layer found in the model")
        return
    
    print(f"Found first Conv2D layer: {first_conv.name}")
    print(f"  Filters: {first_conv.filters}")
    print(f"  Kernel size: {first_conv.kernel_size}")
    
    # Try to get input shape from layer input or model
    try:
        if hasattr(first_conv.input, 'shape'):
            input_shape = first_conv.input.shape
        elif hasattr(first_conv, 'input_shape'):
            input_shape = first_conv.input_shape
        else:
            input_shape = model.input_shape
        print(f"  Input shape: {input_shape}")
    except:
        print(f"  Input shape: (28, 28, 1) [default for digit classifier]")
    
    # Get the weights (filters)
    weights, biases = first_conv.get_weights()
    
    # weights shape: (kernel_height, kernel_width, input_channels, num_filters)
    print(f"\nFilter weights shape: {weights.shape}")
    
    # Determine how many filters to show
    num_filters_total = weights.shape[3]
    if num_filters is None:
        num_filters_to_show = num_filters_total
    else:
        num_filters_to_show = min(num_filters, num_filters_total)
    
    print(f"Displaying {num_filters_to_show} out of {num_filters_total} filters")
    
    # For greyscale input (1 channel), reshape filters for visualization
    if weights.shape[2] == 1:
        # Shape: (kernel_h, kernel_w, 1, num_filters) -> (kernel_h, kernel_w, num_filters)
        filters = weights.squeeze(axis=2)
        filters = np.transpose(filters, (2, 0, 1))  # (num_filters, kernel_h, kernel_w)
    else:
        # For RGB input, show each channel separately or convert to grayscale
        # Taking mean across channels for simplicity
        filters = np.mean(weights, axis=2)
        filters = np.transpose(filters, (2, 0, 1))
    
    # Calculate grid dimensions
    cols = 8
    rows = (num_filters_to_show + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Normalize filters to 0-1 range for visualization
    for i in range(num_filters_to_show):
        filter_img = filters[i]
        
        # Normalize to 0-1 range
        filter_min = filter_img.min()
        filter_max = filter_img.max()
        if filter_max > filter_min:
            filter_normalized = (filter_img - filter_min) / (filter_max - filter_min)
        else:
            filter_normalized = filter_img
        
        # Display
        axes[i].imshow(filter_normalized, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Filter {i}', fontsize=8)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_filters_to_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'First Conv2D Layer Filters\n({first_conv.name}: {num_filters_total} filters, kernel {first_conv.kernel_size})', 
                 fontsize=14)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    else:
        plt.show()
    
    # Print statistics
    print("\nFilter Statistics:")
    print(f"  Min weight value: {weights.min():.4f}")
    print(f"  Max weight value: {weights.max():.4f}")
    print(f"  Mean weight value: {weights.mean():.4f}")
    print(f"  Std weight value: {weights.std():.4f}")
    
    # Print filter weights as numbers
    print("\n" + "="*80)
    print("FILTER WEIGHTS (as numbers):")
    print("="*80)
    
    # Extract filters for display
    if weights.shape[2] == 1:
        filters_for_print = weights.squeeze(axis=2)  # (h, w, num_filters)
    else:
        filters_for_print = np.mean(weights, axis=2)  # Average across channels
    
    for filter_idx in range(min(num_filters_to_show, filters_for_print.shape[2])):
        filter_weights = filters_for_print[:, :, filter_idx]
        print(f"\nFilter {filter_idx} ({first_conv.kernel_size[0]}x{first_conv.kernel_size[1]}):")
        print("-" * 40)
        
        # Print as a matrix with formatting
        for row in filter_weights:
            row_str = "  ".join([f"{val:7.4f}" for val in row])
            print(f"  {row_str}")
        
        # Print statistics for this filter
        print(f"  Min: {filter_weights.min():.4f}, Max: {filter_weights.max():.4f}, Mean: {filter_weights.mean():.4f}")
    
    if num_filters_total > num_filters_to_show:
        print(f"\n... (showing {num_filters_to_show} of {num_filters_total} filters)")
        print("Use -n flag to see more filters, or remove it to see all")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize filters from the first Conv2D layer of a trained model"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained .keras model file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for visualization image (optional, shows interactively if not provided)"
    )
    parser.add_argument(
        "-n", "--num-filters",
        type=int,
        default=None,
        help="Number of filters to display (default: all)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    visualize_first_conv_filters(args.model_path, args.output, args.num_filters)


if __name__ == "__main__":
    main()

