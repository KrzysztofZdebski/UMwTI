"""Debug the projection profile to understand why line detection fails."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path

from config import OUTPUT_DIR, SMOOTHING_KERNEL, MIN_LINE_HEIGHT

pages_dir = OUTPUT_DIR / "1_pages"
preproc_dir = OUTPUT_DIR / "2_preprocessed"

# Test on the 3 preprocessed pages
for pp_file in sorted(preproc_dir.iterdir()):
    print(f"\n{'='*60}")
    print(f"Page: {pp_file.name}")
    print(f"{'='*60}")
    
    binary = cv2.imread(str(pp_file), cv2.IMREAD_GRAYSCALE)
    ink = (binary < 128).astype(np.float64)
    raw_proj = np.sum(ink, axis=1)
    
    # Smooth
    kernel = np.ones(SMOOTHING_KERNEL) / SMOOTHING_KERNEL
    proj = np.convolve(raw_proj, kernel, mode="same")
    
    max_p = np.max(proj)
    mean_p = np.mean(proj)
    
    print(f"  Image size: {binary.shape}")
    print(f"  Projection max: {max_p:.0f}")
    print(f"  Projection mean: {mean_p:.0f}")
    print(f"  Projection min: {np.min(proj):.0f}")
    
    # Test different prominence levels
    inverted = max_p - proj
    print(f"\n  --- find_peaks on inverted profile ---")
    for prom_pct in [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]:
        prom = max_p * prom_pct
        peaks, props = find_peaks(inverted, distance=MIN_LINE_HEIGHT, prominence=prom)
        print(f"  prominence={prom_pct:.0%} ({prom:.0f}): found {len(peaks)} valleys")
        if len(peaks) > 0 and len(peaks) <= 40:
            # Show valley positions and depths
            depths = props['prominences']
            print(f"    Valley positions: {peaks[:10].tolist()}")
            print(f"    Valley depths: {[f'{d:.0f}' for d in depths[:10]]}")
    
    # Also look at the raw projection profile around expected line positions
    print(f"\n  --- Projection sample at regular intervals ---")
    step = binary.shape[0] // 30
    for i in range(0, binary.shape[0], step):
        bar = "█" * int(proj[i] / max_p * 40) if max_p > 0 else ""
        print(f"    row {i:4d}: {proj[i]:6.0f} {bar}")
