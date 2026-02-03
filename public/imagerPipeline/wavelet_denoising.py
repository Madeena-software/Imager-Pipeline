"""
2D Wavelet Transform for X-ray Image Denoising and Background Removal
Uses multi-level wavelet decomposition for superior noise reduction
"""

import cv2
import numpy as np
from pathlib import Path
import warnings

try:
    import pywt

    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("PyWavelets not installed. Install with: pip install PyWavelets")


class WaveletDenoiser:
    """Wavelet-based denoising for X-ray images."""

    def __init__(self, wavelet="db4", level=None):
        """
        Initialize wavelet denoiser.

        Args:
            wavelet: Wavelet type ('db4', 'db8', 'sym4', 'coif1', 'bior4.4')
            level: Decomposition level (None = auto)
        """
        if not PYWT_AVAILABLE:
            raise ImportError(
                "PyWavelets required. Install with: pip install PyWavelets"
            )

        self.wavelet = wavelet
        self.level = level

    def denoise_wavelet(self, image, method="BayesShrink", mode="soft", debug=False):
        """
        Denoise image using wavelet transform.

        Args:
            image: Input grayscale image (float32 [0,1], uint8, or uint16)
            method: 'BayesShrink', 'VisuShrink', or 'manual'
            mode: 'soft' or 'hard' thresholding
            debug: Enable debug output (default: False)

        Returns:
            Denoised image (same type and range as input)
        """
        # Handle different input types
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Check if already normalized to [0, 1] or still in [0, 255/65535] range
            if image.max() > 1.0:
                # Still in uint range, needs normalization
                if image.max() > 255:
                    # Likely uint16 range
                    if debug:
                        print(
                            f"  [DEBUG] Float32 image in uint16 range, normalizing..."
                        )
                    image_norm = image.astype(np.float64) / 65535.0
                else:
                    # Likely uint8 range
                    if debug:
                        print(f"  [DEBUG] Float32 image in uint8 range, normalizing...")
                    image_norm = image.astype(np.float64) / 255.0
                is_float_input = True
                is_16bit = False
            else:
                # Already normalized
                image_norm = image.astype(np.float64)
                is_float_input = True
                is_16bit = False
        else:
            is_float_input = False
            is_16bit = image.dtype == np.uint16

            # Normalize to [0, 1] for wavelet processing
            if is_16bit:
                image_norm = image.astype(np.float64) / 65535.0
            else:
                image_norm = image.astype(np.float64) / 255.0

        # Determine decomposition level if not specified
        if self.level is None:
            max_level = pywt.dwt_max_level(min(image_norm.shape), self.wavelet)
            level = min(max_level, 3)  # Cap at 3 levels to preserve details
        else:
            level = self.level

        if debug:
            print(
                f"  → Wavelet: {self.wavelet}, Level: {level}, Method: {method}, Mode: {mode}"
            )

        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(image_norm, self.wavelet, level=level)

        # Estimate noise from finest scale (highest frequency)
        # Using Median Absolute Deviation (MAD)
        sigma = self._estimate_noise(coeffs[1][0])

        # Threshold coefficients (suppress warnings from pywt)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            coeffs_thresh = self._threshold_coeffs(coeffs, sigma, method, mode)

        # DEBUG: Check thresholded coefficients
        if debug:
            print(
                f"  [DEBUG] Approximation coeffs - min: {coeffs_thresh[0].min():.6f}, max: {coeffs_thresh[0].max():.6f}"
            )
            print(
                f"  [DEBUG] Approximation coeffs - has NaN: {np.isnan(coeffs_thresh[0]).any()}, has inf: {np.isinf(coeffs_thresh[0]).any()}"
            )

        # Reconstruct image
        denoised_norm = pywt.waverec2(coeffs_thresh, self.wavelet)

        # DEBUG: Check reconstruction result
        if debug:
            print(
                f"  [DEBUG] After waverec2 - min: {denoised_norm.min():.6f}, max: {denoised_norm.max():.6f}"
            )
            print(
                f"  [DEBUG] After waverec2 - has NaN: {np.isnan(denoised_norm).any()}, has inf: {np.isinf(denoised_norm).any()}"
            )
            if np.isinf(denoised_norm).any():
                inf_count = np.isinf(denoised_norm).sum()
                print(
                    f"  [WARNING] Found {inf_count} inf values in wavelet reconstruction!"
                )

        # Handle size mismatch due to wavelet decomposition
        denoised_norm = denoised_norm[: image_norm.shape[0], : image_norm.shape[1]]

        # Clip to valid range BEFORE checking for NaN/inf
        denoised_norm = np.clip(denoised_norm, 0, 1)

        # DEBUG: Check after clipping
        if debug:
            print(
                f"  [DEBUG] After clip - min: {denoised_norm.min():.6f}, max: {denoised_norm.max():.6f}"
            )
            print(
                f"  [DEBUG] After clip - has NaN: {np.isnan(denoised_norm).any()}, has inf: {np.isinf(denoised_norm).any()}"
            )

        # Replace any remaining NaN or inf values
        if np.isnan(denoised_norm).any() or np.isinf(denoised_norm).any():
            if debug:
                print(f"  [WARNING] Replacing NaN/inf values in denoised image")
            denoised_norm = np.nan_to_num(
                denoised_norm, nan=0.0, posinf=1.0, neginf=0.0
            )

        # Return in same format as input
        if is_float_input:
            return denoised_norm.astype(np.float32)
        elif is_16bit:
            denoised = (denoised_norm * 65535).astype(np.uint16)
        else:
            denoised = (denoised_norm * 255).astype(np.uint8)

        return denoised

    def _estimate_noise(self, detail_coeffs):
        """Estimate noise level using MAD (Median Absolute Deviation)."""
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745
        return sigma

    def _threshold_coeffs(self, coeffs, sigma, method, mode):
        """Apply thresholding to wavelet coefficients."""
        coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients

        for i in range(1, len(coeffs)):
            # Each level has 3 detail components (cH, cV, cD)
            detail = list(coeffs[i])

            for j in range(3):
                if method == "VisuShrink":
                    # Universal threshold
                    n = detail[j].size
                    threshold = sigma * np.sqrt(2 * np.log(n))

                elif method == "BayesShrink":
                    # Adaptive threshold based on signal variance (softer to preserve details)
                    var_y = np.var(detail[j])
                    var_x = max(var_y - sigma**2, 0)
                    if var_x > 0:
                        threshold = sigma**2 / (np.sqrt(var_x) + 1e-10)
                        threshold = (
                            threshold * 0.5
                        )  # Reduce threshold to keep more details
                    else:
                        threshold = sigma * 0.5

                else:  # manual
                    threshold = 3 * sigma

                # Apply thresholding
                if mode == "soft":
                    detail[j] = pywt.threshold(detail[j], threshold, mode="soft")
                else:
                    detail[j] = pywt.threshold(detail[j], threshold, mode="hard")

            coeffs_thresh.append(tuple(detail))

        return coeffs_thresh

    def multilevel_denoise(self, image, levels=[3, 4, 5]):
        """
        Apply multi-level wavelet denoising and combine results.

        Args:
            image: Input image
            levels: List of decomposition levels to try

        Returns:
            Combined denoised image
        """
        results = []

        for lvl in levels:
            original_level = self.level
            self.level = lvl
            denoised = self.denoise_wavelet(image, method="BayesShrink", mode="soft")
            results.append(denoised.astype(np.float32))
            self.level = original_level

        # Average the results
        combined = np.mean(results, axis=0)

        if image.dtype == np.uint16:
            combined = combined.astype(np.uint16)
        else:
            combined = combined.astype(np.uint8)

        return combined


class WaveletBackgroundRemover:
    """Use wavelet transform to separate background from anatomy."""

    def __init__(self, wavelet="db4"):
        if not PYWT_AVAILABLE:
            raise ImportError(
                "PyWavelets required. Install with: pip install PyWavelets"
            )

        self.wavelet = wavelet

    def remove_background_wavelet(self, image, level=2):
        """
        Remove background using wavelet-based approach.

        Args:
            image: Input grayscale image
            level: Decomposition level (lower = preserve more detail)

        Returns:
            Tuple of (result, mask)
        """
        is_16bit = image.dtype == np.uint16

        # Work directly on original image for mask creation (preserve full resolution)
        if is_16bit:
            image_8bit = (image / 256).astype(np.uint8)
        else:
            image_8bit = image

        print(f"  → Background removal: level={level} (preserving high-freq details)")

        # Create mask from original image (not downsampled)
        # Apply strong blur to get low-frequency approximation
        blurred = cv2.GaussianBlur(image_8bit, (15, 15), 0)

        # Apply Otsu thresholding
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Minimal morphological operations to preserve edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Keep largest component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask_clean = np.zeros_like(mask)
            cv2.drawContours(mask_clean, [largest], -1, 255, -1)
            mask = mask_clean

        # Apply mask using binary multiplication (preserves full resolution)
        mask_binary = (mask > 0).astype(np.float32)

        if is_16bit:
            result = (image.astype(np.float32) * mask_binary).astype(np.uint16)
        else:
            result = (image.astype(np.float32) * mask_binary).astype(np.uint8)

        return result, mask


def process_with_wavelet(
    input_path,
    output_dir,
    wavelet="db4",
    method="BayesShrink",
    denoise_level=None,
    background_level=2,
):
    """
    Complete wavelet-based processing pipeline.

    Args:
        input_path: Path to input image
        output_dir: Output directory
        wavelet: Wavelet type ('db4', 'db8', 'sym4', 'coif1', 'bior4.4')
        method: Denoising method ('BayesShrink', 'VisuShrink', 'manual')
        denoise_level: Decomposition level for denoising (None=auto)
        background_level: Decomposition level for background removal

    Returns:
        Dict with output paths
    """
    print(f"\nProcessing: {Path(input_path).name}")
    print(f"Wavelet: {wavelet}, Method: {method}")
    print("=" * 60)

    # Read image
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Image type: {image.dtype}\n")

    # Step 1: Wavelet denoising
    print("[Step 1] Wavelet denoising...")
    denoiser = WaveletDenoiser(wavelet=wavelet, level=denoise_level)
    denoised = denoiser.denoise_wavelet(image, method=method, mode="soft")

    # Step 2: Wavelet-based background removal
    print("\n[Step 2] Wavelet background removal...")
    bg_remover = WaveletBackgroundRemover(wavelet=wavelet)
    result, mask = bg_remover.remove_background_wavelet(
        denoised, level=background_level
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = Path(input_path).stem

    orig_file = output_path / f"{filename}_0_original.tiff"
    denoise_file = output_path / f"{filename}_1_wavelet_denoised.tiff"
    mask_file = output_path / f"{filename}_2_mask.png"
    result_file = output_path / f"{filename}_3_final.tiff"

    cv2.imwrite(str(orig_file), image)
    cv2.imwrite(str(denoise_file), denoised)
    cv2.imwrite(str(mask_file), mask)
    cv2.imwrite(str(result_file), result)

    # Create comparison
    if image.dtype == np.uint16:
        img_vis = (image / 256).astype(np.uint8)
        denoise_vis = (denoised / 256).astype(np.uint8)
        result_vis = (result / 256).astype(np.uint8)
    else:
        img_vis = image
        denoise_vis = denoised
        result_vis = result

    # Resize for comparison if needed
    max_width = 800
    if img_vis.shape[1] > max_width:
        scale = max_width / img_vis.shape[1]
        new_size = (max_width, int(img_vis.shape[0] * scale))
        img_vis = cv2.resize(img_vis, new_size)
        denoise_vis = cv2.resize(denoise_vis, new_size)
        result_vis = cv2.resize(result_vis, new_size)
        mask_vis = cv2.resize(mask, new_size)
    else:
        mask_vis = mask

    comparison = np.hstack([img_vis, denoise_vis, result_vis, mask_vis])
    comp_file = output_path / f"{filename}_4_comparison.png"
    cv2.imwrite(str(comp_file), comparison)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Original: {orig_file.name}")
    print(f"  Denoised: {denoise_file.name}")
    print(f"  Mask: {mask_file.name}")
    print(f"  Final: {result_file.name}")
    print(f"  Comparison: {comp_file.name}")
    print(f"\nSaved to: {output_path}")
    print("=" * 60)

    return {
        "original": str(orig_file),
        "denoised": str(denoise_file),
        "mask": str(mask_file),
        "final": str(result_file),
        "comparison": str(comp_file),
    }


if __name__ == "__main__":
    import sys

    if not PYWT_AVAILABLE:
        print("ERROR: PyWavelets not installed")
        print("Install with: pip install PyWavelets")
        sys.exit(1)

    if len(sys.argv) < 2:
        print(
            "Usage: python wavelet_denoising.py <image_path> [output_dir] [wavelet] [method]"
        )
        print("\nWavelets: db4 (default), db8, sym4, coif1, bior4.4")
        print("Methods: BayesShrink (default), VisuShrink, manual")
        print("\nExample:")
        print("  python wavelet_denoising.py image.tiff Output db4 BayesShrink")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "Wavelet_Results"
    wavelet = sys.argv[3] if len(sys.argv) > 3 else "db4"
    method = sys.argv[4] if len(sys.argv) > 4 else "BayesShrink"

    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    process_with_wavelet(input_path, output_dir, wavelet=wavelet, method=method)
