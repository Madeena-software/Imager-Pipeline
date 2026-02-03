"""
Image Processing Module for PyScript Browser Environment
Calls the imager pipeline functions for X-ray image processing.

Processing Pipeline:
1. Wavelet Denoising (dark, gain, raw images)
2. Crop and Rotate by detector type
3. Flat-Field Correction (FFC)
4. Auto Thresholding
5. Invert
6. Enhance Contrast (ImageJ-style)
7. CLAHE (optional)
"""

import numpy as np
from js import console
from pyodide.ffi import create_proxy, to_js
import base64
import io

# PyScript/Pyodide compatible imports
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    console.log("PIL not available, using basic image handling")

# Import from imagerPipeline modules
# Try both direct import (PyScript fetched files) and folder import (local files)
try:
    from imagerPipeline.wavelet_denoising import WaveletDenoiser

    WAVELET_AVAILABLE = True
    console.log("✓ WaveletDenoiser imported from imagerPipeline/wavelet_denoising.py")
except ImportError:
    try:
        from wavelet_denoising import WaveletDenoiser

        WAVELET_AVAILABLE = True
        console.log("✓ WaveletDenoiser imported from wavelet_denoising.py")
    except ImportError as e:
        WAVELET_AVAILABLE = False
        console.log(f"✗ wavelet_denoising not available: {e}")

try:
    from imagerPipeline.imagej_replicator import ImageJReplicator

    IMAGEJ_AVAILABLE = True
    console.log("✓ ImageJReplicator imported from imagerPipeline/imagej_replicator.py")
except ImportError:
    try:
        from imagej_replicator import ImageJReplicator

        IMAGEJ_AVAILABLE = True
        console.log("✓ ImageJReplicator imported from imagej_replicator.py")
    except ImportError as e:
        IMAGEJ_AVAILABLE = False
        console.log(f"✗ imagej_replicator not available: {e}")

try:
    from imagerPipeline.complete_pipeline import (
        flat_field_correction,
        crop_and_rotate_by_detector,
        detect_detector_type,
        auto_threshold_detection,
        apply_threshold_separation,
        invert_image,
        denoise_wavelet,
        save_histogram,
    )

    PIPELINE_AVAILABLE = True
    console.log(
        "✓ Pipeline functions imported from imagerPipeline/complete_pipeline.py"
    )
except ImportError:
    try:
        from complete_pipeline import (
            flat_field_correction,
            crop_and_rotate_by_detector,
            detect_detector_type,
            auto_threshold_detection,
            apply_threshold_separation,
            invert_image,
            denoise_wavelet,
            save_histogram,
        )

        PIPELINE_AVAILABLE = True
        console.log("✓ Pipeline functions imported from complete_pipeline.py")
    except ImportError as e:
        PIPELINE_AVAILABLE = False
        console.log(f"✗ complete_pipeline not available: {e}")

try:
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    console.log("SciPy not available, using fallback methods")


class ImageProcessor:
    """Main image processing class for the X-ray pipeline - uses imported pipeline modules."""

    def __init__(self):
        self.denoiser = (
            WaveletDenoiser(wavelet="sym4", level=3) if WAVELET_AVAILABLE else None
        )

    def load_image_from_base64(self, base64_data):
        """Load image from base64 string."""
        # Remove data URL prefix if present
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]

        image_bytes = base64.b64decode(base64_data)

        if PIL_AVAILABLE:
            img = Image.open(io.BytesIO(image_bytes))
            return np.array(img)
        else:
            # Fallback: simple raw loading
            return np.frombuffer(image_bytes, dtype=np.uint8)

    def image_to_base64(self, image, format="PNG"):
        """Convert numpy array to base64 string."""
        if PIL_AVAILABLE:
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Convert float to uint16
                image = (image * 65535).clip(0, 65535).astype(np.uint16)

            img = Image.fromarray(image)
            buffer = io.BytesIO()
            img.save(buffer, format=format)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            return base64.b64encode(image.tobytes()).decode("utf-8")

    def process_pipeline(self, raw_image, dark_image, gain_image, params):
        """
        Execute the complete image processing pipeline using imported modules.

        Args:
            raw_image: Raw radiograph image
            dark_image: Dark calibration image
            gain_image: Gain/flat field image
            params: Dictionary of processing parameters

        Returns:
            Processed image and processing log
        """
        log = []

        # Get parameters with defaults (matching process_single_image in complete_pipeline.py)
        detector_type = params.get("detector_type", "auto")
        wavelet_enabled = params.get("wavelet_enabled", True)
        wavelet_type = params.get("wavelet_type", "sym4")
        wavelet_level = params.get("wavelet_level", 3)
        wavelet_method = params.get("wavelet_method", "BayesShrink")
        threshold_method = params.get("threshold_method", "auto")
        debug_enabled = params.get("debug_enabled", False)
        crop_top = params.get("crop_top", 200)
        crop_bottom = params.get("crop_bottom", 200)
        crop_left = params.get("crop_left", 0)
        crop_right = params.get("crop_right", 0)
        invert_enabled = params.get("invert_enabled", True)
        contrast_enabled = params.get("contrast_enabled", True)
        saturated_pixels = params.get("saturated_pixels", 5)
        normalize = params.get("normalize", True)
        equalize = params.get("equalize", True)
        classic_equalization = params.get("classic_equalization", False)
        clahe_enabled = params.get("clahe_enabled", True)
        clahe_blocksize = params.get("clahe_blocksize", 127)
        clahe_histogram_bins = params.get("clahe_histogram_bins", 256)
        clahe_max_slope = params.get("clahe_max_slope", 0.6)
        clahe_fast = params.get("clahe_fast", False)
        clahe_composite = params.get("clahe_composite", True)

        log.append(f"Starting pipeline processing...")
        log.append(
            f"Pipeline modules: WAVELET={WAVELET_AVAILABLE}, IMAGEJ={IMAGEJ_AVAILABLE}, PIPELINE={PIPELINE_AVAILABLE}"
        )
        log.append(f"Raw shape: {raw_image.shape}, dtype: {raw_image.dtype}")

        # Save raw histogram (step 0)
        if debug_enabled and PIPELINE_AVAILABLE:
            save_histogram(
                raw_image,
                "histogram_raw.png",
                "Raw Image Histogram",
                debug_enabled=True,
            )

        # Auto-detect detector type if needed
        if detector_type == "auto":
            detector_type = "BED"  # Default to BED

        # Convert to float32 [0,1] range
        if raw_image.dtype == np.uint16:
            raw_f = raw_image.astype(np.float32) / 65535.0
            dark_f = dark_image.astype(np.float32) / 65535.0
            gain_f = gain_image.astype(np.float32) / 65535.0
        elif raw_image.dtype == np.uint8:
            raw_f = raw_image.astype(np.float32) / 255.0
            dark_f = dark_image.astype(np.float32) / 255.0
            gain_f = gain_image.astype(np.float32) / 255.0
        else:
            raw_f = raw_image.astype(np.float32)
            dark_f = dark_image.astype(np.float32)
            gain_f = gain_image.astype(np.float32)

        # Step 1: Wavelet Denoising (always enabled to match complete_pipeline.py)
        log.append(
            f"[1/8] Denoising (wavelet={wavelet_type}, level={wavelet_level}, method={wavelet_method}, mode=soft)"
        )

        # Debug: Check input before denoising
        if debug_enabled:
            console.log(
                f"[DEBUG] Before denoise - Raw range: {raw_f.min():.6f} - {raw_f.max():.6f}"
            )
            console.log(
                f"[DEBUG] Before denoise - Raw has NaN: {np.isnan(raw_f).any()}"
            )
            console.log(
                f"[DEBUG] Before denoise - Raw has inf: {np.isinf(raw_f).any()}"
            )

        if PIPELINE_AVAILABLE:
            dark_f = denoise_wavelet(
                dark_f, wavelet_type, wavelet_level, wavelet_method, "soft"
            )
            gain_f = denoise_wavelet(
                gain_f, wavelet_type, wavelet_level, wavelet_method, "soft"
            )
            raw_f = denoise_wavelet(
                raw_f, wavelet_type, wavelet_level, wavelet_method, "soft"
            )

            # Debug: Check output after denoising
            if debug_enabled:
                console.log(
                    f"[DEBUG] After denoise - Raw range: {raw_f.min():.6f} - {raw_f.max():.6f}"
                )
                console.log(
                    f"[DEBUG] After denoise - Raw has NaN: {np.isnan(raw_f).any()}"
                )
                console.log(
                    f"[DEBUG] After denoise - Raw has inf: {np.isinf(raw_f).any()}"
                )
                console.log(
                    f"[DEBUG] After denoise - Raw unique values count: {len(np.unique(raw_f))}"
                )

                # Check if all values are the same (indicates a problem)
                if len(np.unique(raw_f)) < 10:
                    console.log(
                        f"[WARNING] Denoised image has very few unique values! Unique values: {np.unique(raw_f)[:10]}"
                    )

        elif WAVELET_AVAILABLE and self.denoiser:
            self.denoiser.wavelet = wavelet_type
            self.denoiser.level = wavelet_level
            dark_f = self.denoiser.denoise_wavelet(
                dark_f, method=wavelet_method, mode="soft", debug=debug_enabled
            )
            gain_f = self.denoiser.denoise_wavelet(
                gain_f, method=wavelet_method, mode="soft", debug=debug_enabled
            )
            raw_f = self.denoiser.denoise_wavelet(
                raw_f, method=wavelet_method, mode="soft", debug=debug_enabled
            )
        else:
            log.append("  Warning: Wavelet denoising not available, skipping")

        # Validate denoising results - if all values are same or invalid, revert to original
        if PIPELINE_AVAILABLE or (WAVELET_AVAILABLE and self.denoiser):
            if (
                np.isnan(raw_f).any()
                or np.isinf(raw_f).any()
                or len(np.unique(raw_f)) < 10
            ):
                if debug_enabled:
                    console.log(
                        "[ERROR] Denoising produced invalid results! Reverting to original normalized images."
                    )
                log.append("  ERROR: Denoising failed, using original images")
                # Revert to pre-denoising versions
                if raw_image.dtype == np.uint16:
                    raw_f = raw_image.astype(np.float32) / 65535.0
                    dark_f = dark_image.astype(np.float32) / 65535.0
                    gain_f = gain_image.astype(np.float32) / 65535.0
                elif raw_image.dtype == np.uint8:
                    raw_f = raw_image.astype(np.float32) / 255.0
                    dark_f = dark_image.astype(np.float32) / 255.0
                    gain_f = gain_image.astype(np.float32) / 255.0
                else:
                    raw_f = raw_image.astype(np.float32)
                    dark_f = dark_image.astype(np.float32)
                    gain_f = gain_image.astype(np.float32)

        # Save denoised histogram (step 1)
        if debug_enabled and PIPELINE_AVAILABLE:
            save_histogram(
                raw_f,
                "histogram_denoised.png",
                "Denoised Raw Histogram",
                debug_enabled=True,
            )

        # Step 2: Crop and Rotate (using imported function)
        log.append(
            f"[2/8] Crop and Rotate (detector={detector_type}, top={crop_top}, bottom={crop_bottom}, left={crop_left}, right={crop_right})"
        )
        if PIPELINE_AVAILABLE:
            dark_f = crop_and_rotate_by_detector(
                dark_f, detector_type, crop_top, crop_bottom, crop_left, crop_right
            )
            gain_f = crop_and_rotate_by_detector(
                gain_f, detector_type, crop_top, crop_bottom, crop_left, crop_right
            )
            raw_f = crop_and_rotate_by_detector(
                raw_f, detector_type, crop_top, crop_bottom, crop_left, crop_right
            )
        else:
            # Fallback crop and rotate
            h, w = raw_f.shape[0], raw_f.shape[1]
            dark_f = dark_f[crop_top : h - crop_bottom, crop_left : w - crop_right]
            gain_f = gain_f[crop_top : h - crop_bottom, crop_left : w - crop_right]
            raw_f = raw_f[crop_top : h - crop_bottom, crop_left : w - crop_right]
            if detector_type == "TRX":
                dark_f = np.rot90(dark_f, k=1)
                gain_f = np.rot90(gain_f, k=1)
                raw_f = np.rot90(raw_f, k=1)

        # Save cropped histogram (step 2)
        if debug_enabled and PIPELINE_AVAILABLE:
            save_histogram(
                raw_f,
                "histogram_cropped.png",
                "Cropped Raw Histogram",
                debug_enabled=True,
            )

        # Step 3: Flat-Field Correction (using imported function)
        log.append("[3/8] Flat-Field Correction")
        if PIPELINE_AVAILABLE:
            ffc_result = flat_field_correction(raw_f, dark_f, gain_f)
        else:
            # Fallback FFC
            flat_minus_dark = np.maximum(0, gain_f - dark_f)
            mean_value = np.mean(flat_minus_dark)
            raw_minus_dark = np.maximum(0, raw_f - dark_f)
            ffc_result = np.zeros_like(raw_minus_dark)
            mask = flat_minus_dark != 0
            ffc_result[mask] = raw_minus_dark[mask] / flat_minus_dark[mask]
            ffc_result = ffc_result * mean_value
            ffc_result = np.clip(ffc_result, 0, None).astype(np.float32)
        log.append(f"  FFC range: {ffc_result.min():.4f} - {ffc_result.max():.4f}")

        # Save FFC histogram (step 3)
        if debug_enabled and PIPELINE_AVAILABLE:
            save_histogram(
                ffc_result,
                "histogram_ffc.png",
                "FFC Result Histogram",
                debug_enabled=True,
            )

        # Step 4: Auto Thresholding (using imported function)
        log.append(f"[4/8] Auto Thresholding (method={threshold_method})")
        if PIPELINE_AVAILABLE:
            threshold = auto_threshold_detection(
                ffc_result,
                filename=None,
                debug_enabled=debug_enabled,
                threshold_method=threshold_method,
            )
            threshold_result = apply_threshold_separation(ffc_result, threshold)
        else:
            # Fallback threshold using Otsu
            threshold = np.percentile(ffc_result, 25)
            content_mask = ffc_result <= threshold
            content_min = (
                ffc_result[content_mask].min()
                if content_mask.any()
                else ffc_result.min()
            )
            content_max = (
                ffc_result[content_mask].max()
                if content_mask.any()
                else ffc_result.max()
            )
            if content_max > content_min:
                threshold_result = (ffc_result - content_min) / (
                    content_max - content_min
                )
            else:
                threshold_result = ffc_result
            threshold_result = np.where(content_mask, threshold_result, 1.0).astype(
                np.float32
            )
        log.append(f"  Detected threshold: {threshold:.4f}")

        # Save thresholded histogram (step 4)
        if debug_enabled and PIPELINE_AVAILABLE:
            save_histogram(
                threshold_result,
                "histogram_thresholded.png",
                "Thresholded Result Histogram",
                debug_enabled=True,
            )

        # Step 5: Invert (conditionally based on invert_enabled parameter)
        if invert_enabled:
            log.append("[5/8] Inverting image")
            if PIPELINE_AVAILABLE:
                inverted = invert_image(threshold_result)
            else:
                inverted = 1.0 - threshold_result
        else:
            log.append("[5/8] Skipping inversion (disabled)")
            inverted = threshold_result

        # Save inverted histogram (step 5)
        if debug_enabled and PIPELINE_AVAILABLE:
            save_histogram(
                inverted,
                "histogram_inverted.png",
                "Inverted Result Histogram",
                debug_enabled=True,
            )

        # Step 6: Enhance Contrast using ImageJ Replicator (matching complete_pipeline.py)
        if not contrast_enabled:
            log.append("[6/8] Skipping contrast enhancement (disabled)")
            enhanced_uint16 = (inverted * 65535).clip(0, 65535).astype(np.uint16)
        elif not IMAGEJ_AVAILABLE:
            log.append(
                f"[6/8] Enhancing contrast (ImageJ method: saturated={saturated_pixels}%, normalize={normalize}, equalize={equalize})"
            )
            log.append(
                "  Warning: ImageJ processing not available, skipping contrast enhancement"
            )
            enhanced_uint16 = (inverted * 65535).clip(0, 65535).astype(np.uint16)
        else:
            log.append(
                f"[6/8] Enhancing contrast (ImageJ method: saturated={saturated_pixels}%, normalize={normalize}, equalize={equalize})"
            )
            # Convert float32 [0,1] to uint16 for ImageJ processing
            inverted_uint16 = (inverted * 65535).clip(0, 65535).astype(np.uint16)

            # Apply ImageJ-style contrast enhancement
            enhanced = ImageJReplicator.enhance_contrast(
                inverted_uint16,
                saturated_pixels=saturated_pixels,
                normalize=normalize,
                equalize=equalize,
                classic_equalization=classic_equalization,
            )

            # Convert back to uint16 if needed (enhance_contrast returns uint8 by default)
            if enhanced.dtype == np.uint8:
                enhanced_uint16 = (enhanced.astype(np.float32) / 255.0 * 65535).astype(
                    np.uint16
                )
            else:
                enhanced_uint16 = enhanced

            log.append(
                f"  Output range: {enhanced_uint16.min()} - {enhanced_uint16.max()}"
            )

        # Save enhanced histogram (step 6)
        if debug_enabled and PIPELINE_AVAILABLE:
            save_histogram(
                enhanced_uint16,
                "histogram_enhanced.png",
                "Enhanced Result Histogram",
                debug_enabled=True,
            )

        # Step 7: Apply CLAHE using ImageJ Replicator (matching complete_pipeline.py)
        # Parameter guide (ImageJ CLAHE style):
        #   blocksize: 127 = default ImageJ (127 pixels tile)
        #              63  = smaller tiles (more local detail)
        #              255 = larger tiles (more global/smooth)
        #   histogram_bins: 256 = default (full 8-bit range)
        #   max_slope: 1.0-2.0 = kontras ringan (untuk X-ray medis)
        #              3.0     = default ImageJ
        #              4.0+    = kontras kuat
        if not clahe_enabled:
            log.append("[7/8] Skipping CLAHE (disabled)")
            final_result_uint16 = enhanced_uint16
        elif not IMAGEJ_AVAILABLE:
            log.append("[7/8] Applying CLAHE")
            log.append("  Warning: ImageJ processing not available, skipping CLAHE")
            final_result_uint16 = enhanced_uint16
        else:
            log.append("[7/8] Applying CLAHE")
            # Apply CLAHE using ImageJ-style parameters
            clahe_result = ImageJReplicator.apply_clahe(
                enhanced_uint16,
                blocksize=clahe_blocksize,
                histogram_bins=clahe_histogram_bins,
                max_slope=clahe_max_slope,
                mask=None,
                fast=clahe_fast,
                composite=clahe_composite,
            )

            # Convert to uint16 if needed
            if clahe_result.dtype == np.uint8:
                final_result_uint16 = (
                    clahe_result.astype(np.float32) / 255.0 * 65535
                ).astype(np.uint16)
            else:
                final_result_uint16 = clahe_result

            log.append(
                f"  Final output range: {final_result_uint16.min()} - {final_result_uint16.max()}"
            )

        # Save final CLAHE histogram (step 7)
        if debug_enabled and PIPELINE_AVAILABLE:
            save_histogram(
                final_result_uint16,
                "histogram_clahe.png",
                "Final CLAHE Result Histogram",
                debug_enabled=True,
            )

        # Final result is kept as uint16 (matching complete_pipeline.py)
        log.append(
            f"[8/8] Pipeline complete. Output shape: {final_result_uint16.shape}, dtype: {final_result_uint16.dtype}"
        )

        return final_result_uint16, log

    def process_batch(
        self, raw_images, dark_images, gain_images, params, progress_callback=None
    ):
        """
        Process multiple images through the pipeline.

        Args:
            raw_images: List of raw images
            dark_images: List of dark images (same count as raw)
            gain_images: List of gain images (same count as raw)
            params: Processing parameters
            progress_callback: Optional callback for progress updates

        Returns:
            List of processed images and combined log
        """
        results = []
        all_logs = []

        total = len(raw_images)
        for i, (raw, dark, gain) in enumerate(
            zip(raw_images, dark_images, gain_images)
        ):
            if progress_callback:
                progress_callback(i + 1, total)

            result, log = self.process_pipeline(raw, dark, gain, params)
            results.append(result)
            all_logs.extend([f"--- Image {i+1}/{total} ---"] + log)

        return results, all_logs


# Global processor instance
processor = ImageProcessor()


def process_images(dark_base64, gain_base64, raw_base64, params_dict):
    """
    Main entry point for PyScript to process images.

    Args:
        dark_base64: Base64 encoded dark image
        gain_base64: Base64 encoded gain/flat image
        raw_base64: Base64 encoded raw radiograph (the actual X-ray image)
        params_dict: JavaScript object with processing parameters

    Returns:
        Dictionary with processed image and log
    """
    try:
        console.log("Loading images...")

        # Load images with validation
        try:
            dark_img = processor.load_image_from_base64(dark_base64)
            console.log(
                f"✓ Dark image loaded: shape={dark_img.shape}, dtype={dark_img.dtype}"
            )
        except Exception as e:
            console.log(f"✗ Failed to load dark image: {str(e)}")
            raise ValueError(f"Failed to load dark image: {str(e)}")

        try:
            gain_img = processor.load_image_from_base64(gain_base64)
            console.log(
                f"✓ Gain image loaded: shape={gain_img.shape}, dtype={gain_img.dtype}"
            )
        except Exception as e:
            console.log(f"✗ Failed to load gain image: {str(e)}")
            raise ValueError(f"Failed to load gain image: {str(e)}")

        try:
            raw_img = processor.load_image_from_base64(raw_base64)
            console.log(
                f"✓ Raw image loaded: shape={raw_img.shape}, dtype={raw_img.dtype}"
            )
        except Exception as e:
            console.log(f"✗ Failed to load raw image: {str(e)}")
            raise ValueError(f"Failed to load raw image: {str(e)}")

        # Validate image shapes match
        if dark_img.shape != gain_img.shape or dark_img.shape != raw_img.shape:
            error_msg = f"Image shape mismatch! Dark: {dark_img.shape}, Gain: {gain_img.shape}, Raw: {raw_img.shape}"
            console.log(f"✗ {error_msg}")
            raise ValueError(error_msg)

        console.log("✓ All images loaded and validated successfully")

        # Convert JS object to Python dict
        params = dict(params_dict)
        if params.get("debug_enabled", False):
            console.log(
                f"Processing with params: debug_enabled={params.get('debug_enabled', False)}, threshold_method={params.get('threshold_method', 'auto')}"
            )

        # Process
        result, log = processor.process_pipeline(raw_img, dark_img, gain_img, params)

        # Convert result to base64
        result_base64 = processor.image_to_base64(result)

        if params.get("debug_enabled", False):
            console.log("✓ Processing complete successfully")

        return {
            "success": True,
            "image": result_base64,
            "log": "\n".join(log),
            "shape": result.shape,
        }
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        console.log(f"✗ Error during processing: {str(e)}")
        console.log(f"✗ Full traceback:\n{error_details}")
        return {
            "success": False,
            "error": str(e),
            "log": f"Error: {str(e)}\n\nDetails:\n{error_details}",
        }


def get_default_params():
    """Return default processing parameters (matching process_single_image in complete_pipeline.py)."""
    return {
        "detector_type": "auto",
        "wavelet_enabled": True,
        "wavelet_type": "sym4",
        "wavelet_level": 3,
        "wavelet_method": "BayesShrink",
        "threshold_method": "auto",
        "crop_top": 200,
        "crop_bottom": 200,
        "crop_left": 0,
        "crop_right": 0,
        "invert_enabled": True,
        "contrast_enabled": True,
        "saturated_pixels": 5,
        "normalize": True,
        "equalize": True,
        "classic_equalization": False,
        "clahe_enabled": True,
        "clahe_blocksize": 127,
        "clahe_histogram_bins": 256,
        "clahe_max_slope": 0.6,
        "clahe_fast": False,
        "clahe_composite": True,
    }
