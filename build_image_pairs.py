"""
Build pairs of (raw, dark, gain) images for complete pipeline processing.

Scans the 06082024 folder structure and matches raw images with their
corresponding dark and gain calibration images based on:
- Detector type (BED/TRX) detected from anatomy keywords
- Acquisition parameters (kVp, mA, exposure time) from JSON files
"""

import os
import json
import re
from pathlib import Path


def detect_detector_type(filename):
    """
    Detect detector type from filename based on anatomy keywords.
    
    Args:
        filename: Image filename
    
    Returns:
        'TRX' for thorax/chest, 'BED' for other anatomical regions, None if unknown
    """
    filename_lower = filename.lower()
    
    # TRX detector keywords (thorax/chest)
    trx_keywords = ['thorax', 'cervical', 'clavikula', 'humeri', 'humerus', 'clavicula']
    if any(keyword in filename_lower for keyword in trx_keywords):
        return 'TRX'
    
    # BED detector for everything else
    bed_keywords = ['manus', 'cruris', 'pedis', 'antebrachi', 'antebrachii', 'artebrachii', 'pelvis', 'femur', 'genu', 'ankle', 'angkle']
    if any(keyword in filename_lower for keyword in bed_keywords):
        return 'BED'
    
    return None


def parse_filename_params(filename):
    """
    Parse acquisition parameters from filename.
    Format: ... 90kV40mA0,50s ...
    
    Args:
        filename: Image filename
    
    Returns:
        Dict with kVp, mA, exposure_time or None if parsing fails
    """
    import re
    
    # Pattern: 90kV40mA0,50s or 80kV50mA0,32s
    pattern = r'(\d+)kV(\d+)mA(\d+)[,.](\d+)s'
    match = re.search(pattern, filename, re.IGNORECASE)
    
    if match:
        kvp = int(match.group(1))
        ma = int(match.group(2))
        # Combine integer and decimal parts for exposure time
        exp_int = match.group(3)
        exp_dec = match.group(4)
        exposure_time = float(f"{exp_int}.{exp_dec}")
        
        return {
            'kvp': kvp,
            'ma': ma,
            'exposure_time': exposure_time
        }
    
    return None


def parse_json_params(json_path):
    """
    Parse acquisition parameters from JSON file.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        Dict with kVp, mA, exposure_time or None if file doesn't exist
    """
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return {
            'kvp': int(data.get('KVP', 0)),
            'ma': int(data.get('TubeCurrent', 0)),
            'exposure_time': float(data.get('ExposureTime', 0))
        }
    except Exception as e:
        print(f"Warning: Could not parse {json_path}: {e}")
        return None


def find_matching_gain(detector_type, kvp, ma, exposure_time, gain_base_path):
    """
    Find matching gain image based on parameters.
    
    Args:
        detector_type: 'BED' or 'TRX'
        kvp: kVp value
        ma: mA value
        exposure_time: Exposure time in seconds
        gain_base_path: Base path to gain images folder
    
    Returns:
        Path to matching gain file or None if not found
    """
    if detector_type not in ['BED', 'TRX']:
        return None
    
    # Build gain folder path
    gain_folder = os.path.join(gain_base_path, detector_type)
    
    if not os.path.exists(gain_folder):
        return None
    
    # Expected filename format: 90_40_0,50.tiff (kVp_mA_exposure)
    # Convert exposure time to string format (e.g., 0.5 -> "0,50")
    exposure_str = f"{exposure_time:.2f}".replace('.', ',')
    
    # Try exact match
    gain_filename = f"{kvp}_{ma}_{exposure_str}.tiff"
    gain_path = os.path.join(gain_folder, gain_filename)
    
    if os.path.exists(gain_path):
        return gain_path
    
    # Try finding closest match with same kVp and mA
    for filename in os.listdir(gain_folder):
        if not filename.endswith('.tiff'):
            continue
        
        # Parse filename
        match = re.match(r'(\d+)_(\d+)_([\d,]+)\.tiff', filename)
        if match:
            file_kvp = int(match.group(1))
            file_ma = int(match.group(2))
            
            if file_kvp == kvp and file_ma == ma:
                return os.path.join(gain_folder, filename)
    
    # If no exact match, try same kVp only
    for filename in os.listdir(gain_folder):
        if not filename.endswith('.tiff'):
            continue
        
        match = re.match(r'(\d+)_(\d+)_([\d,]+)\.tiff', filename)
        if match:
            file_kvp = int(match.group(1))
            
            if file_kvp == kvp:
                return os.path.join(gain_folder, filename)
    
    return None



def build_image_pairs(base_path):
    """
    Build list of (raw, dark, gain) tuples by grouping files with same 17-char prefix.
    Args:
        base_path: Path to folder containing images
    Returns:
        List of tuples (raw_path, dark_path, gain_path, prefix)
    """
    pairs = []
    skipped = []
    # Scan all files in base_path
    files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
    # Build set of unique 17-char prefixes
    prefixes = set()
    for f in files:
        if len(f) >= 22 and (f.endswith('.tiff') or f.endswith('.tiff.tif')):
            prefixes.add(f[:17])
    # For each prefix, find dark, gain, raw
    for prefix in prefixes:
        dark = gain = raw = None
        for f in files:
            if f.startswith(prefix):
                # Accept both .tiff and .tiff.tif endings
                if f.endswith('_dark.tiff') or f.endswith('_dark.tiff.tif'):
                    dark = os.path.join(base_path, f)
                elif f.endswith('_gain.tiff') or f.endswith('_gain.tiff.tif'):
                    gain = os.path.join(base_path, f)
                elif f.endswith('_rad.tiff') or f.endswith('_rad.tiff.tif'):
                    raw = os.path.join(base_path, f)
        if raw and dark and gain:
            pairs.append((raw, dark, gain, prefix))
        else:
            skipped.append((prefix, f"Missing file(s): raw={bool(raw)}, dark={bool(dark)}, gain={bool(gain)}"))
    return pairs, skipped


def print_summary(pairs, skipped):
    """Print summary of found pairs and skipped images."""
    print(f"\n{'='*80}")
    print(f"IMAGE PAIRING SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully paired: {len(pairs)} images")
    print(f"Skipped: {len(skipped)} images")
    print(f"{'='*80}\n")
    
    if pairs:
        print("Sample paired images:")
        for i, (raw, dark, gain, det) in enumerate(pairs[:5]):
            print(f"\n{i+1}. {os.path.basename(raw)}")
            print(f"   Detector: {str(det)[:3]}")
            print(f"   Dark:     {os.path.basename(dark)}")
            print(f"   Gain:     {os.path.basename(gain)}")
        
        if len(pairs) > 5:
            print(f"\n   ... and {len(pairs) - 5} more images")
    
    if skipped:
        print(f"\n{'='*80}")
        print("Skipped images:")
        print(f"{'='*80}")
        for filename, reason in skipped[:10]:
            print(f"  ✗ {filename}: {reason}")
        
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more skipped images")


def save_pairs_to_file(pairs, output_path='image_pairs.txt'):
    """Save pairs to a text file for reference."""
    with open(output_path, 'w') as f:
        f.write("# Image Pairs for Complete Pipeline Processing\n")
        f.write("# Format: RAW|DARK|GAIN|DETECTOR\n\n")
        
        for raw, dark, gain, det in pairs:
            f.write(f"{raw}|{dark}|{gain}|{str(det)[:3]}\n")
    
    print(f"\nPairs saved to: {output_path}")


def main():
    """Main function to build and display image pairs."""
    base_path = r"D:\RSA\2025-12-09\cropped"
    
    print("Scanning for raw images and matching calibration files...")
    print(f"Base path: {base_path}\n")
    
    pairs, skipped = build_image_pairs(base_path)
    
    print_summary(pairs, skipped)
    
    # Save to file
    if pairs:
        save_pairs_to_file(pairs, 'image_pairs_compile_06082024.txt')
    
    return pairs, skipped


if __name__ == "__main__":
    pairs, skipped = main()
