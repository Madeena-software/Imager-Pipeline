"""
Test script for complete_pipeline.py with automatically paired images.

Uses image pairs from build_image_pairs.py to process medical X-ray images
through the complete pipeline with GPU acceleration and parallel processing.

IMPORTANT: Run this script with the 'grabber' conda environment:
    conda activate grabber
    python test_complete_pipeline.py
"""

import os
import sys
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the complete pipeline module
from complete_pipeline import batch_process_parallel
from build_image_pairs import build_image_pairs


def main():
    """Main test function."""
    print("=" * 80)
    print("TESTING COMPLETE PIPELINE WITH COMPILE 2025-12-09 DATASET")
    print("=" * 80)

    # Read paths from .env file
    base_path = os.getenv("BASE_PATH", r"test")
    output_dir = os.getenv(
        "BATCH_OUTPUT_DIR", os.path.join(base_path, "output_complete_pipeline")
    )

    print(f"\nScanning: {base_path}")
    print(f"Output:   {output_dir}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build pairs
    print("Building image pairs...")
    pairs, skipped = build_image_pairs(base_path)

    if not pairs:
        print("\n✗ No valid image pairs found!")
        return

    print(f"\n{'='*80}")
    print(f"Found {len(pairs)} valid image pairs")
    print(f"Skipped {len(skipped)} images (no JSON or no matching gain)")
    print(f"{'='*80}\n")

    # Check if default option is set in .env
    default_option = os.getenv("DEFAULT_PROCESS_OPTION", "0")

    if default_option != "0":
        choice = default_option
        print(f"Using default option from .env: {choice}")
    else:
        # Ask user how many to process
        print("Options:")
        print("  1. Process first 5 images (quick test)")
        print("  2. Process first 10 images")
        print("  3. Process all images")
        print("  4. Custom number")
        choice = input("Select an option (1-4): ").strip()

    if choice == "1":
        num_images = 5
    elif choice == "2":
        num_images = 10
    elif choice == "3":
        num_images = len(pairs)
    elif choice == "4":
        try:
            num_images = int(input("Enter number of images: "))
            num_images = min(num_images, len(pairs))
        except ValueError:
            print("Invalid number, using 5")
            num_images = 5
    else:
        print("Invalid choice, using 5")
        num_images = 5

    # Select pairs to process
    selected_pairs = pairs[:num_images]

    print(f"\n{'='*80}")
    print(f"Processing {len(selected_pairs)} images")
    print(f"{'='*80}\n")

    # Show what will be processed
    print("Images to process:")
    for i, (raw, dark, gain, det) in enumerate(selected_pairs, 1):
        print(f"  {i}. {os.path.basename(raw)} [{det}]")

    print(f"\nStarting processing...")

    # Process
    start_time = time.time()

    stats = batch_process_parallel(
        selected_pairs, output_dir, num_workers=None  # Auto-detect CPU count
    )

    elapsed = time.time() - start_time

    # Final summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time:     {elapsed:.1f} seconds")
    print(f"Speed:          {len(selected_pairs) / elapsed:.2f} images/second")
    print(f"Success:        {stats['successful']}/{stats['total']}")
    print(f"Failed:         {stats['failed']}")
    print(f"Output folder:  {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
