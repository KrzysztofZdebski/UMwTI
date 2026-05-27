"""Quick diagnostic: preprocess and segment a single PDF page."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from pathlib import Path

from config import OUTPUT_DIR, PREPROCESSED_DIR, LINES_DIR, DEBUG_DIR
from preprocess import preprocess_single
from segment_lines import segment_and_save

# Pick a PDF-extracted page (not the JPGs which are rotated)
pages_dir = OUTPUT_DIR / "1_pages"
pdf_pages = sorted(p for p in pages_dir.glob("*_p001.png"))

if not pdf_pages:
    print("No PDF pages found!")
    sys.exit(1)

# Test on first few PDF pages
test_pages = pdf_pages[:3]
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

for page_path in test_pages:
    print(f"\n--- Processing: {page_path.name} ---")
    
    # Preprocess
    pp_path = preprocess_single(page_path, PREPROCESSED_DIR)
    print(f"  Preprocessed -> {pp_path.name}")
    
    # Check the projection profile
    binary = cv2.imread(str(pp_path), cv2.IMREAD_GRAYSCALE)
    ink = (binary < 128).astype(np.float64)
    projection = np.sum(ink, axis=1)
    
    print(f"  Image shape: {binary.shape}")
    print(f"  Projection max: {projection.max():.0f}")
    print(f"  Projection mean: {projection.mean():.0f}")
    print(f"  Non-zero rows: {np.count_nonzero(projection)}")
    
    # Segment
    line_paths = segment_and_save(pp_path, LINES_DIR, save_debug=True)
    print(f"  Lines found: {len(line_paths)}")

print("\nDone! Check output/debug/ for overlay visualizations.")
