"""Debug just the minima detection on page 2 (which should work)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from scipy.signal import find_peaks
from config import OUTPUT_DIR, SMOOTHING_KERNEL, MIN_LINE_HEIGHT

pp = OUTPUT_DIR / "2_preprocessed" / "20251126183854190_p001.png"
binary = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)
ink = (binary < 128).astype(np.float64)
raw_proj = np.sum(ink, axis=1)
kernel = np.ones(SMOOTHING_KERNEL) / SMOOTHING_KERNEL
proj = np.convolve(raw_proj, kernel, mode="same")

max_p = np.max(proj)
inverted = max_p - proj

# Use same params as in segment_lines.py
prominence = max_p * 0.03
distance = MIN_LINE_HEIGHT  # = 20

peaks, props = find_peaks(inverted, distance=distance, prominence=prominence)
print(f"Found {len(peaks)} valleys")
print(f"Valleys: {peaks.tolist()}")
print(f"Prominences: {[f'{d:.0f}' for d in props['prominences']]}")

# Now trace through find_line_boundaries_minima logic
valleys = sorted(peaks.tolist())
text_threshold = max_p * 0.02
text_rows = np.where(proj > text_threshold)[0]
print(f"\ntext_start={text_rows[0]}, text_end={text_rows[-1]}")

split_points = [text_rows[0]] + valleys + [text_rows[-1]]
print(f"split_points (first 15): {split_points[:15]}")

boundaries = []
for i in range(len(split_points) - 1):
    y_start = split_points[i]
    y_end = split_points[i + 1]
    region_proj = proj[y_start:y_end]
    max_region = np.max(region_proj) if len(region_proj) > 0 else 0
    threshold = max_p * 0.05
    passes = max_region > threshold
    if passes:
        boundaries.append((y_start, y_end))
    # Print first few for debugging
    if i < 5:
        print(f"  Region [{y_start}:{y_end}] max_proj={max_region:.0f} threshold={threshold:.0f} passes={passes}")

print(f"\nFinal boundaries: {len(boundaries)}")
for b in boundaries[:5]:
    print(f"  {b}")
