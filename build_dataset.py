"""
Build a Hugging Face–compatible dataset from segmented line crops.
Creates metadata.csv linking each line image to its (future) ground truth transcription.

This module prepares the dataset structure for both:
  - Inference: just the image paths, run through TrOCR
  - Fine-tuning: image paths + ground truth text columns
"""

import csv
import logging
from pathlib import Path

from config import LINES_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

DATASET_DIR = OUTPUT_DIR / "dataset"
METADATA_FILE = DATASET_DIR / "metadata.csv"


def build_metadata(lines_dir: Path = LINES_DIR,
                   output_dir: Path = DATASET_DIR) -> Path:
    """
    Scan the lines directory and create a metadata CSV with columns:
      - file_name: relative path to the line image (from dataset dir)
      - text: empty string (to be filled with ground truth transcriptions)
      - page: source page identifier
      - line_number: line index within the page

    Returns:
        Path to the metadata CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for page_dir in sorted(lines_dir.iterdir()):
        if not page_dir.is_dir():
            continue

        page_name = page_dir.name
        line_files = sorted(page_dir.glob("line_*.png"))

        for line_file in line_files:
            # Extract line number from filename: line_001.png → 1
            line_num = int(line_file.stem.split("_")[1])

            # Relative path from the lines root
            rel_path = line_file.relative_to(lines_dir)

            rows.append({
                "file_name": str(rel_path).replace("\\", "/"),
                "text": "",
                "page": page_name,
                "line_number": line_num,
            })

    # Write CSV
    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text", "page", "line_number"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Created metadata.csv with %d line entries", len(rows))
    return METADATA_FILE


def print_dataset_stats(lines_dir: Path = LINES_DIR) -> None:
    """Print summary statistics about the extracted dataset."""
    total_lines = 0
    page_count = 0
    lines_per_page = []

    for page_dir in sorted(lines_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        page_count += 1
        n_lines = len(list(page_dir.glob("line_*.png")))
        lines_per_page.append(n_lines)
        total_lines += n_lines

    if not lines_per_page:
        logger.info("No line crops found.")
        return

    import numpy as np
    arr = np.array(lines_per_page)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total pages:       {page_count}")
    print(f"  Total line crops:  {total_lines}")
    print(f"  Lines per page:")
    print(f"    Mean:    {arr.mean():.1f}")
    print(f"    Median:  {np.median(arr):.0f}")
    print(f"    Min:     {arr.min()}")
    print(f"    Max:     {arr.max()}")
    print(f"    Std:     {arr.std():.1f}")
    print("=" * 60 + "\n")
