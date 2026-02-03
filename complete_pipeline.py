import matplotlib.pyplot as plt
import os
from pathlib import Path


# Load environment variables from .env file
def load_env_config():
    """Load configuration from .env file."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    config = {
        # Debug and feature flags
        "DEBUG": False,
        "USE_GPU": False,
        "USE_IMAGEJ": True,
        "USE_DENOISE": True,
        "USE_CROP_ROTATE": True,
        "USE_CLAHE": True,
        "USE_CONTRAST_ENHANCEMENT": True,
        "USE_NORMALIZE": False,
        "USE_INVERT": True,
        # Threshold method
        "THRESHOLD_METHOD": "auto",
        # Wavelet parameters
        "WAVELET_TYPE": "sym4",
        "WAVELET_LEVEL": 3,
        "WAVELET_METHOD": "BayesShrink",
        "WAVELET_MODE": "soft",
        # Cropping parameters
        "CROP_TOP": 0,
        "CROP_BOTTOM": 0,
        "CROP_LEFT": 0,
        "CROP_RIGHT": 0,
        # Contrast enhancement parameters
        "CONTRAST_SATURATED_PIXELS": 5.0,
        "CONTRAST_NORMALIZE": True,
        "CONTRAST_EQUALIZE": True,
        "CONTRAST_CLASSIC_EQUALIZATION": False,
        # CLAHE parameters
        "CLAHE_BLOCKSIZE": 127,
        "CLAHE_HISTOGRAM_BINS": 256,
        "CLAHE_MAX_SLOPE": 0.6,
        "CLAHE_FAST": False,
        "CLAHE_COMPOSITE": True,
        # Normalize (histogram stretch) parameters
        "NORMALIZE_SATURATED_PIXELS": 0.35,
        # Parallel processing
        "NUM_WORKERS": None,
        # Paths
        "RAW_PATH": "",
        "DARK_PATH": "",
        "FLAT_PATH": "",
        "OUTPUT_DIR": "",
    }

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Parse boolean values
                    if key in [
                        "DEBUG",
                        "USE_GPU",
                        "USE_IMAGEJ",
                        "USE_DENOISE",
                        "USE_CROP_ROTATE",
                        "USE_CLAHE",
                        "USE_CONTRAST_ENHANCEMENT",
                        "USE_NORMALIZE",
                        "USE_INVERT",
                        "CONTRAST_NORMALIZE",
                        "CONTRAST_EQUALIZE",
                        "CONTRAST_CLASSIC_EQUALIZATION",
                        "CLAHE_FAST",
                        "CLAHE_COMPOSITE",
                    ]:
                        config[key] = value.lower() in ["1", "true", "yes", "on"]
                    # Parse integer values
                    elif key in [
                        "WAVELET_LEVEL",
                        "CROP_TOP",
                        "CROP_BOTTOM",
                        "CROP_LEFT",
                        "CROP_RIGHT",
                        "CLAHE_BLOCKSIZE",
                        "CLAHE_HISTOGRAM_BINS",
                        "NUM_WORKERS",
                    ]:
                        if value:
                            config[key] = int(value)
                    # Parse float values
                    elif key in [
                        "CLAHE_MAX_SLOPE",
                        "CONTRAST_SATURATED_PIXELS",
                        "NORMALIZE_SATURATED_PIXELS",
                    ]:
                        if value:
                            config[key] = float(value)
                    # String values
                    else:
                        config[key] = value

    return config


# Global config loaded once
CONFIG = load_env_config()


def get_debug_flag():
    return CONFIG["DEBUG"]


def get_use_gpu_flag():
    return CONFIG["USE_GPU"]


def get_use_imagej_flag():
    return CONFIG["USE_IMAGEJ"]


def save_histogram(image, out_path, title=None):
    if get_debug_flag():
        plt.figure(figsize=(8, 4))
        plt.hist(image.ravel(), bins=256, color="blue", alpha=0.7)
        if title:
            plt.title(title)
        plt.xlabel("Pixel Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[DEBUG] Saved histogram: {out_path}")


"""
Complete X-ray Image Processing Pipeline with GPU Acceleration

Processing steps:
1. Denoise dark, gain, raw using wavelet (sym4, level=3, BayesShrink, soft)
2. Crop and rotate by detector type (BED/TRX)
3. Calculate FFC with GPU acceleration
4. Normalize to 16-bit range (scale max value to 65535)
5. Auto Thresholding (background separation)
6. Invert
7. Enhance Contrast like ImageJ (saturated pixels=10%, Normalize, Equalize histogram)
8. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

GPU Acceleration:
- CuPy for array operations (FFC, normalization, thresholding, contrast enhancement)
- Parallel batch processing using multiprocessing
"""

import cv2
import numpy as np
import os
import re
import math
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, cpu_count

# ============================================================================
# CONSTANTS - Bit depth values
# ============================================================================
MAX_8BIT = 255
MAX_16BIT = 65535
MAX_18BIT = 262143
MAX_20BIT = 1048575

# GPU acceleration with CuPy
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np  # Fallback to NumPy

# Check if GPU should be used (must have CuPy AND .env flag enabled)
GPU_AVAILABLE = CUPY_AVAILABLE and get_use_gpu_flag()

if GPU_AVAILABLE:
    print("✓ GPU acceleration enabled (CuPy + USE_GPU=True)")
elif CUPY_AVAILABLE and not get_use_gpu_flag():
    print("✗ GPU acceleration disabled (USE_GPU=False in .env)")
else:
    print("✗ GPU acceleration not available (CuPy not installed)")

# Import wavelet denoising
try:
    from wavelet_denoising import WaveletDenoiser

    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    print(
        "Warning: wavelet_denoising module not available. Wavelet denoising disabled."
    )

# Import ImageJ replicator
try:
    from imagej_replicator import ImageJReplicator

    IMAGEJ_MODULE_AVAILABLE = True
except ImportError:
    IMAGEJ_MODULE_AVAILABLE = False
    print(
        "Warning: imagej_replicator module not available. ImageJ processing disabled."
    )

# Check if ImageJ should be used (must have module AND .env flag enabled)
IMAGEJ_AVAILABLE = IMAGEJ_MODULE_AVAILABLE and get_use_imagej_flag()

if IMAGEJ_AVAILABLE:
    print("✓ ImageJ processing enabled (imagej_replicator + USE_IMAGEJ=True)")
elif IMAGEJ_MODULE_AVAILABLE and not get_use_imagej_flag():
    print("✗ ImageJ processing disabled (USE_IMAGEJ=False in .env)")
else:
    print("✗ ImageJ processing not available (imagej_replicator not installed)")


def denoise_wavelet(image, wavelet="sym4", level=3, method="BayesShrink", mode="soft"):
    """
    Denoise image using wavelet transform.

    Args:
        image: Input image
        wavelet: Wavelet type (default: 'sym4')
        level: Decomposition level (default: 3)
        method: Thresholding method (default: 'BayesShrink')
        mode: Thresholding mode (default: 'soft')

    Returns:
        Denoised image
    """
    if not WAVELET_AVAILABLE:
        print("  Warning: Wavelet denoising not available, returning original image")
        return image

    denoiser = WaveletDenoiser(wavelet=wavelet, level=level)
    return denoiser.denoise_wavelet(image, method=method, mode=mode)


def flat_field_correction(raw_image, dark_image, flat_image):
    """
    Perform flat-field correction on a radiograph image with GPU acceleration.
    Images are already denoised before this step.

    Formula: corrected = (raw - dark) / (flat - dark) * mean(flat - dark)

    Args:
        raw_image: The raw radiograph image to be corrected
        dark_image: The dark frame (image taken with no X-ray exposure)
        flat_image: The flat field image (uniform exposure)

    Returns:
        Flat-field corrected image
    """
    if GPU_AVAILABLE:
        # GPU-accelerated version with CuPy
        raw_32 = cp.asarray(raw_image, dtype=cp.float32)
        dark_32 = cp.asarray(dark_image, dtype=cp.float32)
        flat_32 = cp.asarray(flat_image, dtype=cp.float32)

        # Calculate (flat - dark)
        flat_minus_dark = cp.maximum(0, flat_32 - dark_32)

        # Calculate mean of (flat - dark)
        mean_value = cp.mean(flat_minus_dark)

        # Calculate (raw - dark)
        raw_minus_dark = cp.maximum(0, raw_32 - dark_32)

        # Calculate (raw - dark) / (flat - dark)
        corrected = cp.zeros_like(raw_minus_dark)
        mask = flat_minus_dark != 0
        corrected[mask] = raw_minus_dark[mask] / flat_minus_dark[mask]

        # Multiply by mean to restore intensity scale
        corrected = corrected * mean_value

        # Clip negative values
        corrected = cp.clip(corrected, 0, None)

        # Convert back to CPU and original dtype
        corrected_cpu = cp.asnumpy(corrected)
    else:
        # CPU version with NumPy
        raw_32 = raw_image.astype(np.float32)
        dark_32 = dark_image.astype(np.float32)
        flat_32 = flat_image.astype(np.float32)

        # Calculate (flat - dark)
        flat_minus_dark = np.maximum(0, flat_32 - dark_32)

        # Calculate mean of (flat - dark)
        mean_value = np.mean(flat_minus_dark)

        # Calculate (raw - dark)
        raw_minus_dark = np.maximum(0, raw_32 - dark_32)

        # Calculate (raw - dark) / (flat - dark)
        corrected_cpu = np.zeros_like(raw_minus_dark)
        mask = flat_minus_dark != 0
        corrected_cpu[mask] = raw_minus_dark[mask] / flat_minus_dark[mask]

        # Multiply by mean to restore intensity scale
        corrected_cpu = corrected_cpu * mean_value

        # Clip negative values
        corrected_cpu = np.clip(corrected_cpu, 0, None)

    # Keep as float32 if input is float, otherwise convert back to original dtype
    if raw_image.dtype == np.float32:
        return corrected_cpu.astype(np.float32)
    elif raw_image.dtype == np.uint8:
        corrected_cpu = np.clip(corrected_cpu, 0, MAX_8BIT).astype(np.uint8)
    elif raw_image.dtype == np.uint16:
        corrected_cpu = np.clip(corrected_cpu, 0, MAX_16BIT).astype(np.uint16)
    else:
        corrected_cpu = corrected_cpu.astype(raw_image.dtype)

    return corrected_cpu


def normalize_to_max_value(image, saturated_pixels=None):
    """
    Stretch image histogram to use the full dynamic range using ImageJ's method.

    Uses ImageJ's histogram threshold counting approach (more advanced than percentile)
    to determine optimal stretch range, then applies LUT-based normalization.

    Args:
        image: Input image (uint8 or uint16)
        saturated_pixels: Percentage of pixels to saturate (0-100)
            Default from config: NORMALIZE_SATURATED_PIXELS

    Returns:
        Contrast-stretched image with full dynamic range (same dtype as input)
    """
    if saturated_pixels is None:
        saturated_pixels = CONFIG["NORMALIZE_SATURATED_PIXELS"]

    if not IMAGEJ_AVAILABLE:
        print("  Warning: ImageJ normalization not available, returning original image")
        return image

    # Ensure image is in correct format (uint8 or uint16)
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Convert float to uint16
        image_uint16 = np.clip(image, 0, MAX_16BIT).astype(np.uint16)
    else:
        image_uint16 = image

    # Use ImageJ enhance_contrast with normalize=True, equalize=False
    result = ImageJReplicator.enhance_contrast(
        image_uint16,
        saturated_pixels=saturated_pixels,
        equalize=False,
        normalize=True,
        classic_equalization=False,
    )

    return result


def crop_and_rotate_by_detector(image, detector_type):
    """
    Crop and rotate image based on detector type.
    Crop values are read from .env file (CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT).

    Args:
        image: Input image
        detector_type: 'BED' or 'TRX'

    Returns:
        Cropped and rotated image
    """
    height, width = image.shape[:2]

    # Read crop parameters from config
    crop_top = CONFIG["CROP_TOP"]
    crop_bottom = CONFIG["CROP_BOTTOM"]
    crop_left = CONFIG["CROP_LEFT"]
    crop_right = CONFIG["CROP_RIGHT"]

    # Apply cropping from all sides
    cropped = image[crop_top : height - crop_bottom, crop_left : width - crop_right]

    if detector_type == "TRX":
        # Optionally, keep any TRX-specific rotation
        result = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return result
    else:  # BED
        return cropped


def detect_detector_type(filename):
    """
    Detect detector type from filename.

    Args:
        filename: Image filename

    Returns:
        'TRX' or 'BED'
    """
    filename_upper = filename.upper()

    # TRX detector: Thorax, Humeri, Cervical, Clavikula
    trx_keywords = ["THORAX", "HUMERI", "HUMERUS", "CERVICAL", "CLAVIKULA", "CLAVICULA"]
    if any(keyword in filename_upper for keyword in trx_keywords):
        return "TRX"
    else:
        return "BED"


def auto_threshold_detection(image, filename=None, output_dir=None):
    # Read threshold method from config
    threshold_method = CONFIG.get("THRESHOLD_METHOD", "auto").lower()
    debug_enabled = get_debug_flag()
    """
    Detect optimal threshold for background separation.
    Uses 5 methods with priority on secondary peak (background noise level).
    Updated to match auto_threshold_detection.py and work with float32 [0,1] range.
    
    Args:
        image: Input image (float32 [0,1])
        filename: Optional filename for debug output naming
        output_dir: Optional directory to save debug histogram files
    
    Returns:
        threshold: Optimal threshold value (in same range as image)
    """
    # Calculate histogram with higher resolution (512 bins instead of 256)
    hist, bins = np.histogram(
        image.flatten(), bins=512, range=(image.min(), image.max())
    )
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Smooth histogram
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3)

    # Debug: Show all candidate thresholds and plot them (only if DEBUG=True)
    if debug_enabled:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(bin_centers, hist, label="Histogram", alpha=0.5)
        plt.plot(bin_centers, hist_smooth, label="Smoothed", linewidth=2)

    # Method 1: Percentile (25%)
    threshold_25 = np.percentile(image, 25)
    if debug_enabled:
        plt.axvline(
            threshold_25,
            color="orange",
            linestyle="--",
            label=f"Percentile 25% ({threshold_25:.4f})",
        )

    # Method 2: Valley detection between first two peaks (higher sensitivity: 0.1 prominence)
    peaks, _ = find_peaks(hist_smooth, prominence=hist_smooth.max() * 0.1)
    if len(peaks) >= 2:
        valley_range = (bin_centers >= bin_centers[peaks[0]]) & (
            bin_centers <= bin_centers[peaks[1]]
        )
        if np.any(valley_range):
            valley_idx = np.argmin(hist_smooth[valley_range])
            threshold_valley = bin_centers[valley_range][valley_idx]
        else:
            threshold_valley = threshold_25
    else:
        threshold_valley = threshold_25
    if debug_enabled:
        plt.axvline(
            threshold_valley,
            color="green",
            linestyle="--",
            label=f"Valley ({threshold_valley:.4f})",
        )

    # Method 3: Knee detection (inflection point in CDF)
    cumsum = np.cumsum(hist)
    cumsum_norm = cumsum / cumsum[-1]
    gradient = np.gradient(cumsum_norm)
    gradient_smooth = gaussian_filter1d(gradient, sigma=5)
    second_deriv = np.gradient(gradient_smooth)

    inflection_candidates = np.where(
        np.abs(second_deriv) > np.percentile(np.abs(second_deriv), 90)
    )[0]
    if len(inflection_candidates) > 0:
        threshold_knee = bin_centers[inflection_candidates[0]]
    else:
        threshold_knee = threshold_25
    if debug_enabled:
        plt.axvline(
            threshold_knee,
            color="blue",
            linestyle="--",
            label=f"Knee ({threshold_knee:.4f})",
        )

    # Method 4: Otsu's method (convert to uint16 first for OpenCV)
    if image.dtype == np.float32:
        # Convert to uint16 for Otsu
        image_uint16 = (image * MAX_16BIT).clip(0, MAX_16BIT).astype(np.uint16)
        _, threshold_otsu_uint16 = cv2.threshold(
            image_uint16, 0, MAX_16BIT, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold_otsu = (
            threshold_otsu_uint16 / MAX_16BIT
        )  # Convert back to float32 range
    else:
        _, threshold_otsu = cv2.threshold(
            image, 0, MAX_16BIT, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    # Ensure threshold_otsu is a scalar for formatting/plotting
    if isinstance(threshold_otsu, np.ndarray):
        if threshold_otsu.size == 1:
            threshold_otsu_val = float(threshold_otsu.flat[0])
        else:
            threshold_otsu_val = float(
                threshold_otsu.ravel()[0]
            )  # Take first value if array has multiple elements
    else:
        threshold_otsu_val = float(threshold_otsu)
    if debug_enabled:
        plt.axvline(
            threshold_otsu_val,
            color="purple",
            linestyle="--",
            label=f"Otsu ({threshold_otsu_val:.4f})",
        )

    # Method 5: Secondary peak detection (background noise level)
    threshold_secondary = None
    img_range = image.max() - image.min()
    adaptive_min = image.min() + img_range * 0.40
    adaptive_max = image.min() + img_range * 0.90

    # For float32 range, scale the fixed search values from uint16 range
    if image.dtype == np.float32:
        search_min = min(adaptive_min, 700.0 / MAX_16BIT)
        search_max = max(adaptive_max, min(image.max(), 950.0 / MAX_16BIT))
    else:
        search_min = min(adaptive_min, 700)
        search_max = max(adaptive_max, min(image.max(), 950))

    search_mask = (bin_centers >= search_min) & (bin_centers <= search_max)

    if np.any(search_mask):
        hist_search = hist_smooth[search_mask]
        bins_search = bin_centers[search_mask]

        peaks_secondary, properties = find_peaks(
            hist_search, prominence=hist_smooth.max() * 0.01
        )

        if len(peaks_secondary) > 0:
            most_prominent_idx = np.argmax(properties["prominences"])
            peak_position = bins_search[peaks_secondary[most_prominent_idx]]
            prominence = properties["prominences"][most_prominent_idx]

            # Find valley before secondary peak
            if image.dtype == np.float32:
                valley_search_start = max(peak_position * 0.5, 400.0 / MAX_16BIT)
            else:
                valley_search_start = max(peak_position * 0.5, 400)

            valley_mask = (bin_centers >= valley_search_start) & (
                bin_centers < peak_position
            )
            if np.any(valley_mask):
                hist_before_peak = hist_smooth[valley_mask]
                bins_before_peak = bin_centers[valley_mask]
                valley_idx = np.argmin(hist_before_peak)
                threshold_secondary = bins_before_peak[valley_idx]

    # Results dictionary
    thresholds = {
        "percentile_25": threshold_25,
        "valley": threshold_valley,
        "knee": threshold_knee,
        "otsu": threshold_otsu,
    }

    if threshold_secondary is not None:
        thresholds["secondary_peak"] = threshold_secondary
        if debug_enabled:
            plt.axvline(
                threshold_secondary,
                color="red",
                linestyle="--",
                label=f"Secondary Peak ({threshold_secondary:.4f})",
            )

    # Choose threshold based on .env setting
    # Supported methods: auto, valley, otsu, knee, percentile_25, secondary_peak
    if threshold_method == "valley":
        threshold_auto = threshold_valley
        selected_method = "valley (.env)"
    elif threshold_method == "otsu":
        threshold_auto = threshold_otsu_val
        selected_method = "otsu (.env)"
    elif threshold_method == "knee":
        threshold_auto = threshold_knee
        selected_method = "knee (.env)"
    elif threshold_method == "percentile_25":
        threshold_auto = threshold_25
        selected_method = "percentile_25 (.env)"
    elif threshold_method == "secondary_peak" and threshold_secondary is not None:
        threshold_auto = threshold_secondary
        selected_method = "secondary_peak (.env)"
    elif threshold_method == "auto" or threshold_method is None:
        # Auto mode: intelligent selection based on image characteristics
        # Priority: valley first (if bimodal), then secondary_peak, then otsu, then percentile
        if len(peaks) >= 2:
            # Bimodal histogram: valley is most reliable
            threshold_auto = threshold_valley
            selected_method = "valley (auto)"
        elif threshold_secondary is not None:
            # Use secondary_peak if valley detection failed but secondary peak exists
            threshold_auto = threshold_secondary
            selected_method = "secondary_peak (auto-fallback)"
        elif threshold_otsu_val > 0:
            # Fallback to Otsu for unimodal distributions
            threshold_auto = threshold_otsu_val
            selected_method = "otsu (auto-fallback)"
        else:
            # Ultimate fallback: percentile
            threshold_auto = threshold_25
            selected_method = "percentile_25 (auto-fallback)"
    else:
        # Unknown method in .env, use safe fallback
        if image.dtype == np.float32:
            threshold_auto = 650.0 / MAX_16BIT  # Scale to [0,1] range
        else:
            threshold_auto = 650
        selected_method = f"fixed (unknown method: {threshold_method})"

    # Debug: Print all candidate thresholds and which was selected (only if DEBUG=True)
    if debug_enabled:
        print(f"[DEBUG][THRESHOLDS] Percentile 25%: {threshold_25:.6f}")
        print(f"[DEBUG][THRESHOLDS] Valley: {threshold_valley:.6f}")
        print(f"[DEBUG][THRESHOLDS] Knee: {threshold_knee:.6f}")
        print(f"[DEBUG][THRESHOLDS] Otsu: {threshold_otsu_val:.6f}")
        if threshold_secondary is not None:
            print(f"[DEBUG][THRESHOLDS] Secondary Peak: {threshold_secondary:.6f}")
        print(
            f"[DEBUG][THRESHOLDS] Selected: {threshold_auto:.6f} (method: {selected_method})"
        )

        # Save histogram with all candidate thresholds marked
        plt.axvline(
            threshold_auto,
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"Selected ({threshold_auto:.4f})",
        )
        plt.legend()
        plt.tight_layout()
        # Include filename in debug output to avoid overwriting
        debug_filename = (
            f"debug_histogram_thresholds_{filename}.png"
            if filename
            else "debug_histogram_thresholds.png"
        )
        # Save to output_dir if provided, otherwise save to current directory
        if output_dir:
            debug_filepath = os.path.join(output_dir, debug_filename)
        else:
            debug_filepath = debug_filename
        plt.savefig(debug_filepath)
        plt.close()
        print(f"[DEBUG] Saved threshold histogram: {debug_filepath}")

    return threshold_auto


def apply_threshold_separation(image, threshold):
    """
    Separate content from background and normalize content to full range.
    GPU-accelerated when CuPy is available.
    Works with float32 [0,1] range.

    Args:
        image: Input image (float32 [0,1])
        threshold: Threshold value (in same range as image)

    Returns:
        Processed image with background set to 1.0, content normalized to [0,1]
    """
    if GPU_AVAILABLE:
        # GPU path with CuPy
        img_gpu = cp.asarray(image, dtype=cp.float32)

        # Create mask for content (pixels <= threshold)
        content_mask = img_gpu <= threshold

        # Extract content pixels for min/max calculation
        content_pixels = img_gpu[content_mask]
        if content_pixels.size > 0:
            content_min = float(cp.min(content_pixels))
            content_max = float(cp.max(content_pixels))
        else:
            content_min = float(cp.min(img_gpu))
            content_max = float(cp.max(img_gpu))

        # Normalize content to full range [0, 1]
        if content_max > content_min:
            img_normalized = (img_gpu - content_min) / (content_max - content_min)
        else:
            img_normalized = img_gpu

        # Set background to 1.0 (white), content to normalized values
        result_gpu = cp.where(content_mask, img_normalized, 1.0)
        result_gpu = cp.clip(result_gpu, 0, 1.0)

        # Transfer back to CPU
        result = cp.asnumpy(result_gpu).astype(np.float32)
    else:
        # CPU path with NumPy
        # Create mask for content (pixels <= threshold)
        content_mask = image <= threshold

        # Extract content pixels
        content_only = image.copy()
        content_only[~content_mask] = 0

        # Get min/max of content
        content_pixels = image[content_mask]
        if len(content_pixels) > 0:
            content_min = content_pixels.min()
            content_max = content_pixels.max()
        else:
            content_min = image.min()
            content_max = image.max()

        # Normalize content to full range [0, 1]
        if content_max > content_min:
            content_normalized = (
                (content_only - content_min) / (content_max - content_min)
            ).astype(np.float32)
        else:
            content_normalized = content_only.astype(np.float32)

        # Set background to 1.0 (white)
        result = np.where(content_mask, content_normalized, 1.0).astype(np.float32)

    return result


def invert_image(image):
    """
    Invert the image colors.
    Works with float32 [0,1] range.
    """
    if image.dtype == np.float32:
        return 1.0 - image
    else:
        return cv2.bitwise_not(image)


def process_single_image(
    raw_path, dark_path, flat_path, output_path, detector_type=None
):
    """
    Process a single image through the complete pipeline.

    Pipeline:
    1. Denoise dark, gain, raw using wavelet
    2. Crop and rotate all images (dark, gain, raw) by detector type
    3. Calculate FFC
    4. Auto Thresholding
    5. Invert
    6. Enhance Contrast (ImageJ method: saturated=0.35%, normalize)
    7. CLAHE (ImageJ method: block_size=127, max_slope=1.5)

    Args:
        raw_path: Path to raw image
        dark_path: Path to dark calibration image
        flat_path: Path to flat/gain calibration image
        output_path: Path to save final result
        detector_type: 'BED' or 'TRX' (if None, auto-detect from filename)

    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing: {os.path.basename(raw_path)}")

    # Detect detector type if not provided
    if detector_type is None:
        detector_type = detect_detector_type(os.path.basename(raw_path))
        print(f"  Detected detector: {detector_type}")

    # Load images
    print("  [1/9] Loading images...")
    raw_image = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
    dark_image = cv2.imread(dark_path, cv2.IMREAD_UNCHANGED)
    flat_image = cv2.imread(flat_path, cv2.IMREAD_UNCHANGED)

    if raw_image is None or dark_image is None or flat_image is None:
        print("  ERROR: Failed to load one or more images")
        return False

    print(f"    Raw: {raw_image.shape}, dtype: {raw_image.dtype}")

    # Save histogram for loaded raw image
    debug_dir = os.path.dirname(output_path)
    image_id = os.path.splitext(os.path.basename(raw_path))[0]
    save_histogram(
        raw_image,
        os.path.join(debug_dir, f"histogram_raw_{image_id}.png"),
        title="Raw Image Histogram",
    )

    # Convert to float32 immediately after loading
    raw_image = raw_image.astype(np.float32) / MAX_16BIT
    dark_image = dark_image.astype(np.float32) / MAX_16BIT
    flat_image = flat_image.astype(np.float32) / MAX_16BIT

    # Step 1: Denoise using wavelet (now works with float32 [0,1])
    wavelet_type = CONFIG["WAVELET_TYPE"]
    wavelet_level = CONFIG["WAVELET_LEVEL"]
    wavelet_method = CONFIG["WAVELET_METHOD"]
    wavelet_mode = CONFIG["WAVELET_MODE"]
    print(
        f"  [2/9] Denoising images (wavelet: {wavelet_type}, level={wavelet_level}, {wavelet_method}, {wavelet_mode})..."
    )
    dark_denoised = denoise_wavelet(
        dark_image,
        wavelet=wavelet_type,
        level=wavelet_level,
        method=wavelet_method,
        mode=wavelet_mode,
    )
    flat_denoised = denoise_wavelet(
        flat_image,
        wavelet=wavelet_type,
        level=wavelet_level,
        method=wavelet_method,
        mode=wavelet_mode,
    )
    raw_denoised = denoise_wavelet(
        raw_image,
        wavelet=wavelet_type,
        level=wavelet_level,
        method=wavelet_method,
        mode=wavelet_mode,
    )

    save_histogram(
        raw_denoised,
        os.path.join(debug_dir, f"histogram_denoised_{image_id}.png"),
        title="Denoised Raw Histogram",
    )

    # Step 2: Crop and rotate images - ALL images must be transformed identically
    print(f"  [3/9] Cropping and rotating ({detector_type})...")

    # Apply same transformation to all three images
    dark_cropped = crop_and_rotate_by_detector(dark_denoised, detector_type)
    flat_cropped = crop_and_rotate_by_detector(flat_denoised, detector_type)
    raw_cropped = crop_and_rotate_by_detector(raw_denoised, detector_type)

    save_histogram(
        raw_cropped,
        os.path.join(debug_dir, f"histogram_cropped_{image_id}.png"),
        title="Cropped Raw Histogram",
    )

    crop_info = f"top={CONFIG['CROP_TOP']}, bottom={CONFIG['CROP_BOTTOM']}, left={CONFIG['CROP_LEFT']}, right={CONFIG['CROP_RIGHT']}"
    if detector_type == "TRX":
        print(f"    All images: cropped ({crop_info}), rotated 90° CCW")
    else:
        print(f"    All images: cropped ({crop_info})")

    print(f"    Final shape (all identical): {raw_cropped.shape}")

    # Step 3: FFC with matched dimensions
    print("  [4/9] Applying Flat-Field Correction...")
    ffc_result = flat_field_correction(raw_cropped, dark_cropped, flat_cropped)
    print(f"    FFC output range: {ffc_result.min()} - {ffc_result.max()}")

    save_histogram(
        ffc_result / ffc_result.max() if ffc_result.max() > 0 else ffc_result,
        os.path.join(debug_dir, f"histogram_ffc_{image_id}.png"),
        title="FFC Result Histogram (Normalized 0-1)",
    )

    # Step 4: Normalize to configurable bit depth (optional)
    if CONFIG.get("USE_NORMALIZE", False):
        print(f"  [5/9] Normalizing to max value {MAX_16BIT}...")
        normalized_result = normalize_to_max_value(ffc_result, MAX_16BIT)
        print(
            f"    Normalized range: {normalized_result.min()} - {normalized_result.max()}"
        )

        save_histogram(
            normalized_result / MAX_16BIT,
            os.path.join(debug_dir, f"histogram_normalized_{image_id}.png"),
            title=f"Normalized Result Histogram (max={MAX_16BIT})",
        )
    else:
        print("  [5/9] Normalization skipped (USE_NORMALIZE=False)")
        normalized_result = ffc_result.copy()
        if get_debug_flag():
            print(
                "    [DEBUG] Normalization step was skipped. Passing FFC result forward."
            )

    # Step 5: Auto Thresholding (optional)
    threshold_method = CONFIG.get("THRESHOLD_METHOD", "auto").lower()
    if threshold_method in ["none", "off", "skip", "no"]:
        print("  [6/9] Thresholding skipped (THRESHOLD_METHOD set to 'none'/'off')")
        threshold_result = normalized_result.copy()
        if get_debug_flag():
            print(
                "    [DEBUG] Thresholding step was skipped. Passing normalized result forward."
            )
    else:
        print("  [6/9] Auto Thresholding...")
        threshold = auto_threshold_detection(
            normalized_result, filename=image_id, output_dir=debug_dir
        )
        if get_debug_flag():
            print(f"    [DEBUG] Detected threshold: {threshold:.6f}")
            # Debug: pixel counts below/above threshold
            below = np.count_nonzero(normalized_result <= threshold)
            above = np.count_nonzero(normalized_result > threshold)
            total = normalized_result.size
            print(f"    [DEBUG] Pixels <= threshold: {below} ({below/total:.2%})")
            print(f"    [DEBUG] Pixels > threshold: {above} ({above/total:.2%})")
        threshold_result = apply_threshold_separation(normalized_result, threshold)
        if get_debug_flag():
            print(
                f"    [DEBUG] Thresholded min/max: {threshold_result.min()} - {threshold_result.max()}"
            )
            print(
                f"    [DEBUG] Thresholded nonzero count: {np.count_nonzero(threshold_result)}"
            )

        save_histogram(
            threshold_result,
            os.path.join(debug_dir, f"histogram_thresholded_{image_id}.png"),
            title="Thresholded Result Histogram",
        )

    # Step 6: Invert
    print("  [7/9] Inverting image...")
    inverted = invert_image(threshold_result)

    save_histogram(
        inverted,
        os.path.join(debug_dir, f"histogram_inverted_{image_id}.png"),
        title="Inverted Result Histogram",
    )

    # Step 7: Enhance Contrast using ImageJ Replicator
    print("  [8/9] Enhancing contrast (ImageJ method)")
    if not CONFIG["USE_CONTRAST_ENHANCEMENT"]:
        print("    Skipping contrast enhancement (USE_CONTRAST_ENHANCEMENT=False)")
        enhanced_uint16 = (inverted * MAX_16BIT).clip(0, MAX_16BIT).astype(np.uint16)
    elif not IMAGEJ_AVAILABLE:
        print(
            "    Warning: ImageJ processing not available, skipping contrast enhancement"
        )
        enhanced_uint16 = (inverted * MAX_16BIT).clip(0, MAX_16BIT).astype(np.uint16)
    else:
        # Convert float32 [0,1] to uint16 for ImageJ processing
        inverted_uint16 = (inverted * MAX_16BIT).clip(0, MAX_16BIT).astype(np.uint16)

        # Apply ImageJ-style contrast enhancement
        enhanced = ImageJReplicator.enhance_contrast(
            inverted_uint16,
            saturated_pixels=CONFIG["CONTRAST_SATURATED_PIXELS"],
            normalize=CONFIG["CONTRAST_NORMALIZE"],
            equalize=CONFIG["CONTRAST_EQUALIZE"],
            classic_equalization=CONFIG["CONTRAST_CLASSIC_EQUALIZATION"],
        )

        # Convert back to uint16 if needed (enhance_contrast returns uint8 by default)
        if enhanced.dtype == np.uint8:
            enhanced_uint16 = (
                enhanced.astype(np.float32) / MAX_8BIT * MAX_16BIT
            ).astype(np.uint16)
        else:
            enhanced_uint16 = enhanced

        print(f"    Output range: {enhanced_uint16.min()} - {enhanced_uint16.max()}")

    save_histogram(
        enhanced_uint16,
        os.path.join(debug_dir, f"histogram_enhanced_{image_id}.png"),
        title="Enhanced Result Histogram",
    )

    # Step 7: Apply CLAHE using ImageJ Replicator
    # Parameter guide (ImageJ CLAHE style):
    #   blocksize: 127 = default ImageJ (127 pixels tile)
    #              63  = smaller tiles (more local detail)
    #              255 = larger tiles (more global/smooth)
    #   histogram_bins: 256 = default (full 8-bit range)
    #   max_slope: 1.0-2.0 = kontras ringan (untuk X-ray medis)
    #              3.0     = default ImageJ
    #              4.0+    = kontras kuat
    print("  [9/9] Applying CLAHE")
    if not CONFIG["USE_CLAHE"]:
        print("    Skipping CLAHE (USE_CLAHE=False)")
        final_result_uint16 = enhanced_uint16
    elif not IMAGEJ_AVAILABLE:
        print("    Warning: ImageJ processing not available, skipping CLAHE")
        final_result_uint16 = enhanced_uint16
    else:
        # Apply CLAHE using ImageJ-style parameters
        clahe_result = ImageJReplicator.apply_clahe(
            enhanced_uint16,
            blocksize=CONFIG["CLAHE_BLOCKSIZE"],
            histogram_bins=CONFIG["CLAHE_HISTOGRAM_BINS"],
            max_slope=CONFIG["CLAHE_MAX_SLOPE"],
            mask=None,
            fast=CONFIG["CLAHE_FAST"],
            composite=CONFIG["CLAHE_COMPOSITE"],
        )

        # Convert to uint16 if needed
        if clahe_result.dtype == np.uint8:
            final_result_uint16 = (
                clahe_result.astype(np.float32) / MAX_8BIT * MAX_16BIT
            ).astype(np.uint16)
        else:
            final_result_uint16 = clahe_result

        print(
            f"    Final output range: {final_result_uint16.min()} - {final_result_uint16.max()}"
        )

    save_histogram(
        final_result_uint16,
        os.path.join(debug_dir, f"histogram_clahe_{image_id}.png"),
        title="Final CLAHE Result Histogram",
    )

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_result_uint16)
    print(f"  ✓ Saved to: {output_path}")

    return True


def process_worker(args):
    """
    Worker function for parallel processing.

    Args:
        args: Tuple of (raw_path, dark_path, flat_path, output_path, detector_type)

    Returns:
        Tuple of (success, filename)
    """
    raw_path, dark_path, flat_path, output_path, detector_type = args
    try:
        success = process_single_image(
            raw_path, dark_path, flat_path, output_path, detector_type
        )

        # Clean up GPU memory after each image to prevent memory buildup
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

        return (success, os.path.basename(raw_path))
    except Exception as e:
        print(f"✗ Error processing {os.path.basename(raw_path)}: {str(e)}")

        # Clean up GPU memory on error too
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

        return (False, os.path.basename(raw_path))


def batch_process_parallel(image_list, output_dir, num_workers=None):
    """
    Process multiple images in parallel using multiprocessing.

    Args:
        image_list: List of tuples (raw_path, dark_path, flat_path, detector_type)
        output_dir: Output directory for processed images
        num_workers: Number of parallel workers (default: from .env or auto)

    Returns:
        Statistics dict with success/failure counts
    """
    if num_workers is None:
        # Try to get from config first
        num_workers = CONFIG.get("NUM_WORKERS")

        if num_workers is None:
            # Use fewer workers when GPU is available to avoid GPU memory contention
            # GPU handles internal parallelism more efficiently than multiprocessing
            if GPU_AVAILABLE:
                num_workers = 4  # Optimal for GPU to avoid memory contention
            else:
                num_workers = max(1, cpu_count() - 1)

    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(image_list)} images")
    print(f"Workers: {num_workers} parallel processes")
    print(f"GPU: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")
    print(f"{'='*70}\n")

    # Prepare arguments for workers
    args_list = []
    for raw_path, dark_path, flat_path, detector_type in image_list:
        filename = os.path.basename(raw_path)

        # Use splitext to properly handle file extensions
        name_without_ext, ext = os.path.splitext(filename)

        # Create output filename with _processed suffix (allows tracking of re-processing)
        output_filename = f"{name_without_ext}_processed{ext}"

        output_path = os.path.join(output_dir, output_filename)
        args_list.append((raw_path, dark_path, flat_path, output_path, detector_type))

    # Process in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_worker, args_list)

    # Collect statistics
    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful

    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total images:           {len(results)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed:                 {failed}")
    print(f"Output directory:       {output_dir}")
    print(f"{'='*70}\n")

    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "results": results,
    }


def main():
    """
    Main processing function.
    Configure your input/output paths here.
    """
    print("=" * 70)
    print("COMPLETE X-RAY IMAGE PROCESSING PIPELINE")
    print("=" * 70)
    print("\nProcessing steps:")
    print("  1. Denoise (wavelet: sym4, level=3, BayesShrink, soft)")
    print("  2. Crop & Rotate by detector type:")
    print("      - TRX: crop 200px each side, rotate 90° CCW")
    print("      - BED: crop 200px each side")
    print("  3. Flat-Field Correction (FFC) with GPU acceleration")
    print("  4. Auto Thresholding (background separation)")
    print("  5. Invert")
    print("  6. Enhance Contrast (ImageJ method: saturated=0.35%, normalize=True)")
    print("  7. CLAHE (ImageJ method: block_size=127, max_slope=1.5)")
    print("\nOptimizations:")
    print("  - GPU acceleration for FFC and array operations (CuPy)")
    print("  - Parallel batch processing (multiprocessing)")
    print("=" * 70)

    # Example 1: Single image processing
    # Load paths from .env or use defaults
    raw_path = CONFIG.get("RAW_PATH") or r"test\BED_1765259553954_rad.tiff"
    dark_path = CONFIG.get("DARK_PATH") or r"test\BED_1765259553954_dark.tiff"
    flat_path = CONFIG.get("FLAT_PATH") or r"test\BED_1765259553954_gain.tiff"
    output_dir = CONFIG.get("OUTPUT_DIR") or r"test\output"

    # Ensure paths use raw strings for proper Windows path handling
    raw_path = rf"{raw_path}" if raw_path and "\\" not in raw_path else raw_path
    dark_path = rf"{dark_path}" if dark_path and "\\" not in dark_path else dark_path
    flat_path = rf"{flat_path}" if flat_path and "\\" not in flat_path else flat_path
    output_dir = (
        rf"{output_dir}" if output_dir and "\\" not in output_dir else output_dir
    )

    # Construct output path with proper filename and extension
    raw_filename = os.path.splitext(os.path.basename(raw_path))[0]
    output_path = os.path.join(output_dir, f"{raw_filename}_processed.tiff")
    success = process_single_image(raw_path, dark_path, flat_path, output_path)
    print(f"\nSingle image processing {'succeeded' if success else 'failed'}")

    # Example 2: Batch processing
    # image_list = [
    #     ("raw1.tiff", "dark.tiff", "flat1.tiff", None),  # Auto-detect detector
    #     ("raw2.tiff", "dark.tiff", "flat2.tiff", "TRX"),
    #     ("raw3.tiff", "dark.tiff", "flat3.tiff", "BED"),
    # ]
    # output_dir = "path/to/output_folder"
    # stats = batch_process_parallel(image_list, output_dir, num_workers=8)

    print("\nTo use this script:")
    print("  1. For single image: Call process_single_image()")
    print("  2. For batch: Call batch_process_parallel()")
    print("  3. Uncomment examples in main() and modify paths")


if __name__ == "__main__":
    main()
