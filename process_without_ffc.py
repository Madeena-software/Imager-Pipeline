"""
Image Processing Pipeline without FFC (Flat-Field Correction)

This script processes images through all pipeline steps except FFC:
1. Denoise (wavelet)
2. Crop & Rotate by detector type
3. Normalize (optional)
4. Auto Thresholding (optional)
5. Invert
6. Enhance Contrast (ImageJ method)
7. Apply CLAHE (ImageJ method)

Can process:
- Single image
- Batch process all images in a folder (no pairing required)
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List

# Import functions from complete_pipeline
from complete_pipeline import (
    CONFIG,
    denoise_wavelet,
    crop_and_rotate_by_detector,
    detect_detector_type,
    normalize_to_max_value,
    auto_threshold_detection,
    apply_threshold_separation,
    invert_image,
    save_histogram,
    get_debug_flag,
    MAX_16BIT,
)

# Import ImageJ replicator
from imagej_replicator import ImageJReplicator


def process_single_image_no_ffc(
    input_path: str,
    output_path: str,
    detector_type: Optional[str] = None,
    save_debug: bool = False,
) -> bool:
    """
    Process a single image through the pipeline without FFC.

    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        detector_type: 'BED' or 'TRX' (if None, auto-detect from filename)
        save_debug: Whether to save debug histograms

    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing: {os.path.basename(input_path)}")

    # Detect detector type if not provided
    if detector_type is None:
        detector_type = detect_detector_type(os.path.basename(input_path))
        print(f"  Auto-detected detector type: {detector_type}")

    # Load image
    print("  [1/8] Loading image...")
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"  ✗ Failed to load image: {input_path}")
        return False

    print(f"    Shape: {image.shape}, dtype: {image.dtype}")

    # Setup debug directory if needed
    if save_debug or get_debug_flag():
        debug_dir = os.path.dirname(output_path)
        image_id = os.path.splitext(os.path.basename(input_path))[0]
        save_histogram(
            image,
            os.path.join(debug_dir, f"histogram_raw_{image_id}.png"),
            title="Raw Image Histogram",
        )

    # Convert to float32 [0,1] range
    image = image.astype(np.float32) / MAX_16BIT

    # Step 1: Denoise using wavelet
    if CONFIG.get("USE_DENOISE", True):
        wavelet_type = CONFIG["WAVELET_TYPE"]
        wavelet_level = CONFIG["WAVELET_LEVEL"]
        wavelet_method = CONFIG["WAVELET_METHOD"]
        wavelet_mode = CONFIG["WAVELET_MODE"]
        print(
            f"  [2/8] Denoising (wavelet: {wavelet_type}, level={wavelet_level}, {wavelet_method}, {wavelet_mode})..."
        )
        denoised = denoise_wavelet(
            image,
            wavelet=wavelet_type,
            level=wavelet_level,
            method=wavelet_method,
            mode=wavelet_mode,
        )

        if save_debug or get_debug_flag():
            save_histogram(
                denoised,
                os.path.join(debug_dir, f"histogram_denoised_{image_id}.png"),
                title="Denoised Histogram",
            )
    else:
        print("  [2/8] Skipping denoising (USE_DENOISE=False)")
        denoised = image

    # Step 2: Crop and rotate
    if CONFIG.get("USE_CROP_ROTATE", True):
        print(f"  [3/8] Cropping and rotating ({detector_type})...")
        cropped = crop_and_rotate_by_detector(denoised, detector_type)

        crop_info = f"top={CONFIG['CROP_TOP']}, bottom={CONFIG['CROP_BOTTOM']}, left={CONFIG['CROP_LEFT']}, right={CONFIG['CROP_RIGHT']}"
        if detector_type == "TRX":
            print(f"    TRX: {crop_info}, rotate 90° CCW")
        else:
            print(f"    BED: {crop_info}")
        print(f"    Final shape: {cropped.shape}")

        if save_debug or get_debug_flag():
            save_histogram(
                cropped,
                os.path.join(debug_dir, f"histogram_cropped_{image_id}.png"),
                title="Cropped Histogram",
            )
    else:
        print("  [3/8] Skipping crop and rotate (USE_CROP_ROTATE=False)")
        cropped = denoised

    # Step 3: Normalize (optional)
    if CONFIG.get("USE_NORMALIZE", False):
        print("  [4/8] Normalizing to full range...")
        normalized = normalize_to_max_value(
            (cropped * MAX_16BIT).clip(0, MAX_16BIT).astype(np.uint16),
            saturated_pixels=CONFIG["NORMALIZE_SATURATED_PIXELS"],
        )
        normalized = normalized.astype(np.float32) / MAX_16BIT
        print(f"    Range: {normalized.min():.4f} - {normalized.max():.4f}")

        if save_debug or get_debug_flag():
            save_histogram(
                normalized,
                os.path.join(debug_dir, f"histogram_normalized_{image_id}.png"),
                title="Normalized Histogram",
            )
    else:
        print("  [4/8] Skipping normalization (USE_NORMALIZE=False)")
        normalized = cropped

    # Step 4: Auto Thresholding (optional)
    threshold_method = CONFIG.get("THRESHOLD_METHOD", "auto").lower()
    if threshold_method in ["none", "off", "skip", "no"]:
        print("  [5/8] Skipping auto thresholding (THRESHOLD_METHOD=none)")
        threshold_result = normalized
    else:
        print(f"  [5/8] Auto thresholding (method: {threshold_method})...")
        threshold = auto_threshold_detection(
            normalized,
            filename=os.path.basename(input_path),
            output_dir=(
                os.path.dirname(output_path)
                if (save_debug or get_debug_flag())
                else None
            ),
        )
        print(f"    Threshold: {threshold:.4f}")

        threshold_result = apply_threshold_separation(normalized, threshold)
        print(
            f"    Result range: {threshold_result.min():.4f} - {threshold_result.max():.4f}"
        )

        if save_debug or get_debug_flag():
            save_histogram(
                threshold_result,
                os.path.join(debug_dir, f"histogram_threshold_{image_id}.png"),
                title="Threshold Result Histogram",
            )

    # Step 5: Invert
    if CONFIG.get("USE_INVERT", True):
        print("  [6/8] Inverting image...")
        inverted = invert_image(threshold_result)

        if save_debug or get_debug_flag():
            save_histogram(
                inverted,
                os.path.join(debug_dir, f"histogram_inverted_{image_id}.png"),
                title="Inverted Histogram",
            )
    else:
        print("  [6/8] Skipping invert (USE_INVERT=False)")
        inverted = threshold_result

    # Step 6: Enhance Contrast using ImageJ
    print("  [7/8] Enhancing contrast (ImageJ method)...")
    if not CONFIG["USE_CONTRAST_ENHANCEMENT"]:
        print("    Skipping contrast enhancement (USE_CONTRAST_ENHANCEMENT=False)")
        enhanced_uint16 = (inverted * MAX_16BIT).clip(0, MAX_16BIT).astype(np.uint16)
    else:
        # Convert to uint16 for ImageJ processing
        inverted_uint16 = (inverted * MAX_16BIT).clip(0, MAX_16BIT).astype(np.uint16)

        enhanced_uint16 = ImageJReplicator.enhance_contrast(
            inverted_uint16,
            saturated_pixels=CONFIG["CONTRAST_SATURATED_PIXELS"],
            normalize=CONFIG["CONTRAST_NORMALIZE"],
            equalize=CONFIG["CONTRAST_EQUALIZE"],
            classic_equalization=CONFIG["CONTRAST_CLASSIC_EQUALIZATION"],
        )
        print(f"    Enhanced range: {enhanced_uint16.min()} - {enhanced_uint16.max()}")

    if save_debug or get_debug_flag():
        save_histogram(
            enhanced_uint16,
            os.path.join(debug_dir, f"histogram_enhanced_{image_id}.png"),
            title="Enhanced Histogram",
        )

    # Step 7: Apply CLAHE using ImageJ
    print("  [8/8] Applying CLAHE...")
    if not CONFIG["USE_CLAHE"]:
        print("    Skipping CLAHE (USE_CLAHE=False)")
        final_result_uint16 = enhanced_uint16
    else:
        final_result_uint16 = ImageJReplicator.apply_clahe(
            enhanced_uint16,
            blocksize=CONFIG["CLAHE_BLOCKSIZE"],
            histogram_bins=CONFIG["CLAHE_HISTOGRAM_BINS"],
            max_slope=CONFIG["CLAHE_MAX_SLOPE"],
            fast=CONFIG["CLAHE_FAST"],
            composite=CONFIG["CLAHE_COMPOSITE"],
        )
        print(
            f"    CLAHE range: {final_result_uint16.min()} - {final_result_uint16.max()}"
        )

    if save_debug or get_debug_flag():
        save_histogram(
            final_result_uint16,
            os.path.join(debug_dir, f"histogram_final_{image_id}.png"),
            title="Final Result Histogram",
        )

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_result_uint16)
    print(f"  ✓ Saved to: {output_path}")

    return True


def batch_process_folder(
    input_folder: str,
    output_folder: str,
    detector_type: Optional[str] = None,
    extensions: List[str] = [".tif", ".tiff", ".png", ".jpg"],
    save_debug: bool = False,
) -> dict:
    """
    Batch process all images in a folder.

    Args:
        input_folder: Path to input folder
        output_folder: Path to output folder
        detector_type: 'BED' or 'TRX' (if None, auto-detect for each file)
        extensions: List of file extensions to process
        save_debug: Whether to save debug histograms

    Returns:
        Statistics dict with success/failure counts
    """
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING FOLDER: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*70}\n")

    # Find all image files
    input_path = Path(input_folder)
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"✗ No image files found in {input_folder}")
        print(f"  Looking for extensions: {extensions}")
        return {"total": 0, "successful": 0, "failed": 0}

    print(f"Found {len(image_files)} image(s) to process\n")

    # Process each image
    successful = 0
    failed = 0
    results = []

    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")

        # Construct output path
        output_filename = f"{image_file.stem}_processed{image_file.suffix}"
        output_path = os.path.join(output_folder, output_filename)

        try:
            success = process_single_image_no_ffc(
                str(image_file),
                output_path,
                detector_type=detector_type,
                save_debug=save_debug,
            )

            if success:
                successful += 1
                results.append((True, image_file.name))
            else:
                failed += 1
                results.append((False, image_file.name))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
            results.append((False, image_file.name))

    # Print summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total images:           {len(image_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed:                 {failed}")
    print(f"Output folder:          {output_folder}")
    print(f"{'='*70}\n")

    return {
        "total": len(image_files),
        "successful": successful,
        "failed": failed,
        "results": results,
    }


def main():
    """
    Main function with hardcoded configuration.

    Configure your processing here:
    """
    # ========================================================================
    # CONFIGURATION - Edit these values
    # ========================================================================

    # Processing mode: "single" or "folder"
    MODE = "single"

    # Input path (file path for single mode, folder path for folder mode)
    INPUT_PATH = r"test\BED_1765259553954_rad.tiff"

    # Output path (file path for single mode, folder path for folder mode)
    OUTPUT_PATH = r"test\output\processed.tiff"

    # Optional: Detector type ("BED", "TRX", or None for auto-detect)
    DETECTOR_TYPE = None

    # Optional: Save debug histograms (True/False)
    SAVE_DEBUG = False

    # Optional: File extensions to process (for folder mode)
    FILE_EXTENSIONS = [".tif", ".tiff", ".png", ".jpg"]

    # ========================================================================
    # END CONFIGURATION
    # ========================================================================

    print("=" * 70)
    print("IMAGE PROCESSING PIPELINE (NO FFC)")
    print("=" * 70)
    print("\nProcessing steps:")
    print("  1. Denoise (wavelet)")
    print("  2. Crop & Rotate by detector type")
    print("  3. Normalize (optional)")
    print("  4. Auto Thresholding (optional)")
    print("  5. Invert")
    print("  6. Enhance Contrast (ImageJ method)")
    print("  7. Apply CLAHE (ImageJ method)")
    print("\nNote: FFC (Flat-Field Correction) is skipped")
    print("=" * 70 + "\n")

    # Validate mode
    if MODE.lower() not in ["single", "folder"]:
        print(f"✗ Error: Invalid MODE '{MODE}'. Must be 'single' or 'folder'")
        sys.exit(1)

    # Process based on mode
    if MODE.lower() == "single":
        # Single image mode
        print(f"Mode: Single image processing")
        print(f"Input:  {INPUT_PATH}")
        print(f"Output: {OUTPUT_PATH}\n")

        success = process_single_image_no_ffc(
            INPUT_PATH, OUTPUT_PATH, detector_type=DETECTOR_TYPE, save_debug=SAVE_DEBUG
        )

        if success:
            print("\n✓ Processing completed successfully")
            sys.exit(0)
        else:
            print("\n✗ Processing failed")
            sys.exit(1)

    elif MODE.lower() == "folder":
        # Batch processing mode
        print(f"Mode: Batch folder processing")
        print(f"Input folder:  {INPUT_PATH}")
        print(f"Output folder: {OUTPUT_PATH}\n")

        stats = batch_process_folder(
            INPUT_PATH,
            OUTPUT_PATH,
            detector_type=DETECTOR_TYPE,
            extensions=FILE_EXTENSIONS,
            save_debug=SAVE_DEBUG,
        )

        if stats["failed"] == 0:
            print("\n✓ All images processed successfully")
            sys.exit(0)
        elif stats["successful"] > 0:
            print(f"\n⚠ Completed with {stats['failed']} error(s)")
            sys.exit(1)
        else:
            print("\n✗ All images failed to process")
            sys.exit(1)


if __name__ == "__main__":
    main()
