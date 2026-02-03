"""
Replicator untuk fungsi pemrosesan citra ImageJ di Python.

Termasuk implementasi:
- ContrastEnhancer (Enhance Contrast)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE Implementation Reference:
    Zuiderveld, Karel. "Contrast limited adaptive histogram equalization."
    Graphics gems IV. Academic Press Professional, Inc., 1994. 474-485.

License: GPL v2 (mengikuti lisensi asli ImageJ CLAHE plugin)
"""

import cv2
import numpy as np
import math
from typing import Optional, Tuple, Union
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ThreadPoolExecutor
import warnings

# Konstanta bit depth
MAX_UINT8 = 255
MAX_UINT16 = 65535
BINS_UINT8 = 256
BINS_UINT16 = 65536


class ImageJReplicator:
    """
    Kelas utilitas untuk mereplikasi fungsi pemrosesan citra ImageJ
    di lingkungan Python/OpenCV dengan presisi tinggi.

    Implementasi ini mengikuti logika ContrastEnhancer.java dari ImageJ
    untuk memastikan hasil yang identik.
    """

    @staticmethod
    def _get_min_and_max_imagej(
        histogram: np.ndarray, saturated: float, pixel_count: int
    ) -> Tuple[int, int]:
        """
        Mereplikasi metode getMinAndMax() dari ImageJ ContrastEnhancer.java.

        ImageJ menggunakan pendekatan berbasis histogram dengan threshold counting,
        bukan percentile-based seperti implementasi umum lainnya.

        Args:
            histogram: Array histogram (256 bins untuk 8-bit, 65536 untuk 16-bit)
            saturated: Persentase piksel tersaturasi (0-100)
            pixel_count: Total jumlah piksel dalam gambar

        Returns:
            Tuple (hmin, hmax): Indeks histogram untuk min dan max
        """
        hsize = len(histogram)

        # ImageJ: threshold = (pixelCount * saturated / 200.0)
        # Ini membagi saturated menjadi setengah untuk low dan high
        if saturated > 0.0:
            threshold = int(pixel_count * saturated / 200.0)
        else:
            threshold = 0

        # Cari hmin: scan dari kiri sampai count melebihi threshold
        i = -1
        found = False
        count = 0
        maxindex = hsize - 1

        while not found and i < maxindex:
            i += 1
            count += histogram[i]
            found = count > threshold
        hmin = i

        # Cari hmax: scan dari kanan sampai count melebihi threshold
        i = hsize
        count = 0
        found = False

        while not found and i > 0:
            i -= 1
            count += histogram[i]
            found = count > threshold
        hmax = i

        return hmin, hmax

    @staticmethod
    def _normalize_imagej(
        image: np.ndarray, min_val: float, max_val: float
    ) -> np.ndarray:
        """
        Mereplikasi metode normalize() dari ImageJ ContrastEnhancer.java.

        ImageJ menggunakan LUT (Look-Up Table) untuk normalisasi, yang memberikan
        hasil yang sedikit berbeda dari linear scaling biasa.

        Args:
            image: Citra input grayscale
            min_val: Nilai minimum dari histogram stretching
            max_val: Nilai maksimum dari histogram stretching

        Returns:
            Citra yang dinormalisasi dengan tipe data yang sama
        """
        original_dtype = image.dtype

        # Tentukan range berdasarkan bit depth
        if original_dtype == np.uint16:
            max2 = 65535
            range_val = 65536
        else:
            max2 = 255
            range_val = 256

        # Buat LUT seperti ImageJ
        lut = np.zeros(range_val, dtype=np.float64)

        for i in range(range_val):
            if i <= min_val:
                lut[i] = 0
            elif i >= max_val:
                lut[i] = max2
            else:
                # Formula ImageJ: (int)(((double)(i-min)/(max-min))*max2)
                lut[i] = int(((i - min_val) / (max_val - min_val)) * max2)

        # Terapkan LUT
        lut = lut.astype(original_dtype)
        return lut[image]

    @staticmethod
    def enhance_contrast(
        image: np.ndarray,
        saturated_pixels: float = 0.35,
        equalize: bool = False,
        normalize: bool = True,
        classic_equalization: bool = False,
    ) -> np.ndarray:
        """
        Mereplikasi ImageJ 'Enhance Contrast' (ContrastEnhancer.java).

        Args:
            image (np.ndarray): Citra input (Grayscale atau RGB).
            saturated_pixels (float): Persentase piksel tersaturasi (Default ImageJ: 0.35).
            equalize (bool): Jika True, lakukan Histogram Equalization (varian ImageJ).
            normalize (bool): Jika True, lakukan stretching dengan LUT (Data diubah).
            classic_equalization (bool): Jika True, gunakan HE klasik.
                Jika False (default), gunakan sqrt-weighted HE seperti ImageJ.

        Returns:
            np.ndarray: Citra hasil pemrosesan (preserves input bit depth).

        Raises:
            ValueError: Jika input tidak valid.
            TypeError: Jika tipe data input tidak sesuai.
        """
        # Validasi input
        if image is None:
            raise ValueError("Citra input tidak boleh kosong")

        if not isinstance(image, np.ndarray):
            raise TypeError("Input harus berupa numpy array")

        if image.size == 0:
            raise ValueError("Array citra tidak boleh kosong")

        # Clamp saturated_pixels seperti ImageJ
        if saturated_pixels < 0.0:
            saturated_pixels = 0.0
        if saturated_pixels > 100.0:
            saturated_pixels = 100.0

        # Simpan tipe data asli untuk preservasi bit depth
        original_dtype = image.dtype

        # ---------------------------------------------------------
        # MODE 1: EQUALIZE HISTOGRAM (Varian ImageJ)
        # ---------------------------------------------------------
        if equalize:
            try:
                if len(image.shape) == 3:
                    # Konversi ke LAB, proses luminance channel (L)
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    l_eq = ImageJReplicator._equalize_imagej_variant(
                        l, classic_equalization
                    )
                    res_lab = cv2.merge((l_eq, a, b))
                    return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)
                else:
                    return ImageJReplicator._equalize_imagej_variant(
                        image, classic_equalization
                    )
            except cv2.error as e:
                raise ValueError(f"Gagal melakukan konversi color space: {e}")

        # ---------------------------------------------------------
        # MODE 2: STRETCH HISTOGRAM (stretchHistogram dari ImageJ)
        # ---------------------------------------------------------
        # Jika normalize=False di ImageJ, hanya display range yang berubah (metadata).
        # Di Python, kita mengembalikan gambar asli tanpa perubahan.
        if not normalize:
            return image

        # Proses berdasarkan tipe gambar
        if len(image.shape) == 3:
            # Untuk RGB, proses luminance di LAB space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_stretched = ImageJReplicator._stretch_histogram_imagej(
                l, saturated_pixels, normalize
            )
            res_lab = cv2.merge((l_stretched, a, b))
            return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)
        else:
            return ImageJReplicator._stretch_histogram_imagej(
                image, saturated_pixels, normalize
            )

    @staticmethod
    def _stretch_histogram_imagej(
        image: np.ndarray, saturated: float, normalize: bool = True
    ) -> np.ndarray:
        """
        Implementasi internal stretchHistogram dari ImageJ.

        Mengikuti logika exact dari ContrastEnhancer.java:
        1. Hitung histogram
        2. Cari hmin dan hmax menggunakan threshold counting
        3. Hitung min dan max dari bin positions
        4. Terapkan normalisasi dengan LUT

        Args:
            image: Citra grayscale input
            saturated: Persentase saturasi (0-100)
            normalize: Jika True, terapkan LUT normalisasi

        Returns:
            Citra yang di-stretch
        """
        original_dtype = image.dtype

        # Tentukan bins berdasarkan bit depth
        if original_dtype == np.uint16:
            num_bins = BINS_UINT16
        else:
            num_bins = BINS_UINT8

        # Hitung histogram
        histogram, _ = np.histogram(image.flatten(), bins=num_bins, range=(0, num_bins))
        histogram = histogram.astype(np.int64)

        pixel_count = image.size

        # Dapatkan hmin dan hmax menggunakan metode ImageJ
        hmin, hmax = ImageJReplicator._get_min_and_max_imagej(
            histogram, saturated, pixel_count
        )

        if hmax <= hmin:
            return image  # Tidak ada stretching yang diperlukan

        # Untuk 8-bit dan 16-bit, min dan max langsung dari bin index
        min_val = float(hmin)
        max_val = float(hmax)

        if normalize:
            return ImageJReplicator._normalize_imagej(image, min_val, max_val)
        else:
            # Tanpa normalize, ImageJ hanya mengubah display range
            # Di Python kita kembalikan asli
            return image

    @staticmethod
    def _equalize_imagej_variant(
        gray_image: np.ndarray, classic_equalization: bool = False
    ) -> np.ndarray:
        """
        Implementasi exact Histogram Equalization dari ImageJ ContrastEnhancer.java.

        ImageJ menggunakan integrasi trapesium dengan opsi weighted (sqrt) atau classic.
        Algoritma ini menggunakan formula:
        - sum = getWeightedValue(histogram, 0)
        - for i=1 to max-1: sum += 2 * getWeightedValue(histogram, i)
        - sum += getWeightedValue(histogram, max)
        - scale = range/sum
        - lut[0] = 0, lut[max] = max
        - for i=1 to max-1: lut[i] = round(cumulative_sum * scale)

        Args:
            gray_image: Citra grayscale (uint8 atau uint16)
            classic_equalization: Jika True, gunakan histogram langsung.
                Jika False (default), gunakan sqrt(histogram) untuk hasil lebih halus.

        Returns:
            Citra hasil ekualisasi dengan tipe data yang sama dengan input.
        """
        original_dtype = gray_image.dtype

        # Tentukan range berdasarkan bit depth
        if original_dtype == np.uint16:
            max_val = 65535
            range_val = 65535
        else:
            max_val = 255
            range_val = 255

        # Hitung histogram
        histogram = np.bincount(gray_image.flatten(), minlength=max_val + 1)
        histogram = histogram.astype(np.float64)

        def get_weighted_value(hist: np.ndarray, i: int, classic: bool) -> float:
            """Replikasi getWeightedValue dari ImageJ."""
            h = hist[i]
            if h < 2 or classic:
                return float(h)
            return math.sqrt(float(h))

        # Hitung sum menggunakan formula ImageJ (integrasi trapesium)
        total_sum = get_weighted_value(histogram, 0, classic_equalization)
        for i in range(1, max_val):
            total_sum += 2 * get_weighted_value(histogram, i, classic_equalization)
        total_sum += get_weighted_value(histogram, max_val, classic_equalization)

        # Edge case: jika sum sangat kecil
        if total_sum < 1e-10:
            return gray_image

        scale = range_val / total_sum

        # Buat LUT
        lut = np.zeros(range_val + 1, dtype=np.int64)
        lut[0] = 0

        cumsum = get_weighted_value(histogram, 0, classic_equalization)
        for i in range(1, max_val):
            delta = get_weighted_value(histogram, i, classic_equalization)
            cumsum += delta
            lut[i] = int(round(cumsum * scale))
            cumsum += delta

        lut[max_val] = max_val

        # Clip LUT ke range valid
        lut = np.clip(lut, 0, max_val).astype(original_dtype)

        # Terapkan LUT
        return lut[gray_image]

    # ---------------------------------------------------------
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # ---------------------------------------------------------

    @staticmethod
    def apply_clahe(
        image: np.ndarray,
        blocksize: int = 127,
        histogram_bins: int = 256,
        max_slope: float = 3.0,
        mask: Optional[np.ndarray] = None,
        fast: bool = True,
        composite: bool = True,
    ) -> np.ndarray:
        """
        Menerapkan CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Mereplikasi plugin CLAHE dari ImageJ/Fiji (mpicbg.ij.clahe)
        yang dikembangkan oleh Stephan Saalfeld.

        Reference:
            Zuiderveld, Karel. "Contrast limited adaptive histogram equalization."
            Graphics gems IV. Academic Press Professional, Inc., 1994. 474-485.

        Args:
            image: Input image (grayscale atau RGB)
            blocksize: Ukuran blok dalam pixel (default: 127)
            histogram_bins: Jumlah histogram bins (default: 256)
            max_slope: Maximum slope untuk contrast limiting (default: 3.0)
            mask: Optional mask (ByteProcessor equivalent)
            fast: Gunakan metode cepat yang kurang akurat (default: True)
            composite: Untuk RGB, proses setiap channel terpisah (default: True)

        Returns:
            Processed image dengan tipe data yang sama

        Example:
            >>> result = ImageJReplicator.apply_clahe(image, blocksize=127, histogram_bins=256, max_slope=3.0)
        """
        block_radius = (blocksize - 1) // 2
        bins = histogram_bins - 1

        if fast:
            return ImageJReplicator._clahe_fast(
                image, block_radius, bins, max_slope, mask, composite
            )
        else:
            return ImageJReplicator._clahe_precise(
                image, block_radius, bins, max_slope, mask, composite
            )

    @staticmethod
    def _clahe_create_histogram_lut(
        histogram: np.ndarray, slope: float, bins: int, n_pixels: int, max_val: int
    ) -> np.ndarray:
        """
        Buat LUT dari histogram dengan contrast limiting.
        """
        clip_limit = int(slope * n_pixels / (bins + 1))
        if clip_limit < 1:
            clip_limit = 1

        clipped_hist = histogram.copy().astype(np.float64)
        excess = 0

        for i in range(len(clipped_hist)):
            if clipped_hist[i] > clip_limit:
                excess += clipped_hist[i] - clip_limit
                clipped_hist[i] = clip_limit

        redistribution = excess / (bins + 1)
        clipped_hist += redistribution

        residual = excess - redistribution * (bins + 1)
        if residual > 0:
            step = max(1, (bins + 1) // int(residual + 1))
            for i in range(0, len(clipped_hist), step):
                if residual <= 0:
                    break
                clipped_hist[i] += 1
                residual -= 1

        cdf = np.cumsum(clipped_hist)
        cdf_min = cdf[0]
        cdf_max = cdf[-1]

        if cdf_max - cdf_min > 0:
            lut = ((cdf - cdf_min) / (cdf_max - cdf_min) * max_val).astype(np.uint8)
        else:
            lut = np.arange(bins + 1, dtype=np.uint8)

        return lut

    @staticmethod
    def _clahe_compute_block_histogram(
        image: np.ndarray,
        row: int,
        col: int,
        block_radius: int,
        bins: int,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Hitung histogram untuk blok di sekitar pixel (row, col).
        """
        height, width = image.shape[:2]

        r_min = max(0, row - block_radius)
        r_max = min(height, row + block_radius + 1)
        c_min = max(0, col - block_radius)
        c_max = min(width, col + block_radius + 1)

        block = image[r_min:r_max, c_min:c_max]

        if mask is not None:
            mask_block = mask[r_min:r_max, c_min:c_max]
            block = block[mask_block > 0]

        n_pixels = block.size

        if n_pixels == 0:
            return np.zeros(bins + 1, dtype=np.int64), 0

        if image.dtype == np.uint16:
            quantized = (block.astype(np.float64) / MAX_UINT16 * bins).astype(np.int32)
        else:
            quantized = (block.astype(np.float64) / MAX_UINT8 * bins).astype(np.int32)

        quantized = np.clip(quantized, 0, bins)
        histogram = np.bincount(quantized.flatten(), minlength=bins + 1)

        return histogram.astype(np.int64), n_pixels

    @staticmethod
    def _clahe_fast(
        image: np.ndarray,
        block_radius: int,
        bins: int,
        slope: float,
        mask: Optional[np.ndarray],
        composite: bool,
    ) -> np.ndarray:
        """
        Implementasi CLAHE cepat menggunakan OpenCV sebagai basis.
        """
        original_dtype = image.dtype

        if len(image.shape) == 3:
            if composite:
                channels = cv2.split(image)
                processed = []
                for ch in channels:
                    processed.append(
                        ImageJReplicator._clahe_apply_single(
                            ch, block_radius, bins, slope, mask
                        )
                    )
                return cv2.merge(processed)
            else:
                if original_dtype == np.uint16:
                    img_8bit = (image / 256).astype(np.uint8)
                    lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
                else:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

                l, a, b = cv2.split(lab)
                l_processed = ImageJReplicator._clahe_apply_single(
                    l, block_radius, bins, slope, mask
                )
                result_lab = cv2.merge([l_processed, a, b])
                result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

                if original_dtype == np.uint16:
                    result = result.astype(np.uint16) * 256
                return result
        else:
            return ImageJReplicator._clahe_apply_single(
                image, block_radius, bins, slope, mask
            )

    @staticmethod
    def _clahe_apply_single(
        image: np.ndarray,
        block_radius: int,
        bins: int,
        slope: float,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Terapkan CLAHE ke single-channel image.
        """
        original_dtype = image.dtype

        if original_dtype == np.uint16:
            work_image = image.copy()
        else:
            work_image = image.copy()

        block_size = block_radius * 2 + 1
        height, width = work_image.shape

        tiles_x = max(1, width // block_size)
        tiles_y = max(1, height // block_size)

        clahe_obj = cv2.createCLAHE(clipLimit=slope, tileGridSize=(tiles_x, tiles_y))
        result = clahe_obj.apply(work_image)

        if mask is not None:
            mask_binary = (mask > 0).astype(np.uint8)
            result = np.where(mask_binary, result, work_image)

        return result.astype(original_dtype)

    @staticmethod
    def _clahe_precise(
        image: np.ndarray,
        block_radius: int,
        bins: int,
        slope: float,
        mask: Optional[np.ndarray],
        composite: bool,
    ) -> np.ndarray:
        """
        Implementasi CLAHE presisi tinggi yang lebih dekat ke ImageJ.
        """
        original_dtype = image.dtype

        if len(image.shape) == 3:
            if composite:
                channels = cv2.split(image)
                processed = []
                for ch in channels:
                    processed.append(
                        ImageJReplicator._clahe_apply_precise(
                            ch, block_radius, bins, slope, mask
                        )
                    )
                return cv2.merge(processed)
            else:
                if original_dtype == np.uint16:
                    img_8bit = (image / 256).astype(np.uint8)
                    lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
                else:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

                l, a, b = cv2.split(lab)
                l_processed = ImageJReplicator._clahe_apply_precise(
                    l, block_radius, bins, slope, mask
                )
                result_lab = cv2.merge([l_processed, a, b])
                result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

                if original_dtype == np.uint16:
                    result = result.astype(np.uint16) * 256
                return result
        else:
            return ImageJReplicator._clahe_apply_precise(
                image, block_radius, bins, slope, mask
            )

    @staticmethod
    def _clahe_apply_precise(
        image: np.ndarray,
        block_radius: int,
        bins: int,
        slope: float,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Implementasi CLAHE presisi untuk single channel dengan interpolasi bilinear.
        """
        original_dtype = image.dtype
        height, width = image.shape

        if original_dtype == np.uint16:
            max_val = MAX_UINT16
            scale_factor = MAX_UINT16 / bins
        else:
            max_val = MAX_UINT8
            scale_factor = MAX_UINT8 / bins

        block_size = block_radius * 2 + 1

        n_blocks_y = max(1, (height + block_size - 1) // block_size)
        n_blocks_x = max(1, (width + block_size - 1) // block_size)

        if n_blocks_y > 1:
            step_y = (height - 1) / (n_blocks_y - 1)
        else:
            step_y = height

        if n_blocks_x > 1:
            step_x = (width - 1) / (n_blocks_x - 1)
        else:
            step_x = width

        luts = np.zeros((n_blocks_y, n_blocks_x, bins + 1), dtype=np.float64)

        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                cy = int(by * step_y) if n_blocks_y > 1 else height // 2
                cx = int(bx * step_x) if n_blocks_x > 1 else width // 2

                hist, n_pixels = ImageJReplicator._clahe_compute_block_histogram(
                    image, cy, cx, block_radius, bins, mask
                )

                if n_pixels > 0:
                    luts[by, bx] = ImageJReplicator._clahe_create_histogram_lut(
                        hist, slope, bins, n_pixels, bins
                    )
                else:
                    luts[by, bx] = np.arange(bins + 1)

        result = np.zeros_like(image, dtype=np.float64)
        quantized = (image.astype(np.float64) / max_val * bins).astype(np.int32)
        quantized = np.clip(quantized, 0, bins)

        for y in range(height):
            for x in range(width):
                fy = y / step_y if n_blocks_y > 1 else 0
                fx = x / step_x if n_blocks_x > 1 else 0

                by0 = int(fy)
                bx0 = int(fx)
                by1 = min(by0 + 1, n_blocks_y - 1)
                bx1 = min(bx0 + 1, n_blocks_x - 1)

                wy = fy - by0
                wx = fx - bx0

                pixel = quantized[y, x]

                v00 = luts[by0, bx0, pixel]
                v01 = luts[by0, bx1, pixel]
                v10 = luts[by1, bx0, pixel]
                v11 = luts[by1, bx1, pixel]

                v0 = v00 * (1 - wx) + v01 * wx
                v1 = v10 * (1 - wx) + v11 * wx
                v = v0 * (1 - wy) + v1 * wy

                result[y, x] = v * scale_factor

        if mask is not None:
            mask_binary = mask > 0
            result = np.where(mask_binary, result, image.astype(np.float64))

        result = np.clip(result, 0, max_val).astype(original_dtype)

        return result
