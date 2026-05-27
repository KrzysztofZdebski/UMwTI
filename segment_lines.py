"""
Stage 2–3: Line segmentation using horizontal projection profiles.
Crops individual text lines from preprocessed page images, ready for TrOCR.

Uses a combined approach:
  1. Smooth horizontal projection profile
  2. Find local minima (valleys) in the projection using scipy
  3. Fall back to simple thresholding if scipy approach fails
  4. Merge close lines (diacritics), filter by height
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

from config import (
    DEBUG_DIR,
    LINE_PADDING_X,
    LINE_PADDING_Y,
    LINES_DIR,
    MAX_LINE_HEIGHT,
    MIN_LINE_HEIGHT,
    PREPROCESSED_DIR,
    SMOOTHING_KERNEL,
    VALLEY_THRESHOLD,
    SEGMENTATION_METHOD,
    KRAKEN_MODEL,
)

logger = logging.getLogger(__name__)

# Check if Kraken is available
try:
    from kraken import blla
    HAS_KRAKEN = True
except ImportError:
    HAS_KRAKEN = False


def save_debug_overlay_kraken(image: np.ndarray, lines: list, output_path: Path) -> None:
    """
    Save a visualization with detected baseline polygons and baseline polylines
    overlaid on the page (in green and blue respectively).
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for i, line in enumerate(lines):
        boundary = getattr(line, 'boundary', None) if not isinstance(line, dict) else line.get('boundary', None)
        baseline = getattr(line, 'baseline', None) if not isinstance(line, dict) else line.get('baseline', None)

        # Draw boundary polygon (green)
        if boundary:
            pts = np.array(boundary, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw baseline polyline (blue)
        if baseline:
            pts = np.array(baseline, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=False, color=(255, 0, 0), thickness=2)

        # Add line number label
        if boundary and len(boundary) > 0:
            x0, y0 = boundary[0]
            cv2.putText(vis, str(i + 1), (x0, y0 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(str(output_path), vis)


def compute_horizontal_projection(binary: np.ndarray) -> np.ndarray:
    """
    Compute the horizontal projection profile: for each row, count
    the number of dark (ink) pixels.

    Expects binary image where 0 = ink, 255 = background.
    """
    # Invert: ink becomes 1, background becomes 0
    ink = (binary < 128).astype(np.float64)
    projection = np.sum(ink, axis=1)
    return projection


def smooth_projection(projection: np.ndarray, kernel_size: int = SMOOTHING_KERNEL) -> np.ndarray:
    """
    Smooth the projection profile with a 1D moving average to reduce
    noise from diacritics, crossed t-bars, and scanning artifacts.
    """
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(projection, kernel, mode="same")
    return smoothed


def find_line_boundaries_minima(projection: np.ndarray,
                                min_line_height: int = MIN_LINE_HEIGHT) -> list[tuple[int, int]]:
    """
    Find text line boundaries using local minima (valley) detection.

    Strategy:
        1. Invert the smoothed projection to turn valleys into peaks
        2. Use scipy.signal.find_peaks to locate valleys
        3. Split the signal at these valley points
        4. Each region between valleys that contains significant ink is a text line

    This approach handles dense handwriting where projection never drops
    to a low absolute threshold, but still has relative valleys between lines.
    """
    if np.max(projection) == 0:
        return []

    # Invert the projection: valleys become peaks
    inverted = np.max(projection) - projection

    # Find valleys (peaks in inverted signal)
    # distance = minimum distance between valleys (should be at least a line height)
    # prominence = how much a valley must stand out; use relative prominence
    max_proj = np.max(projection)
    prominence = max_proj * 0.03  # Valley must be at least 3% of max below neighbors
    distance = min_line_height  # At least MIN_LINE_HEIGHT between valleys

    peaks, properties = find_peaks(inverted, distance=distance, prominence=prominence)

    if len(peaks) < 2:
        # Not enough valleys found, fall back to threshold approach
        return find_line_boundaries_threshold(projection)

    # Sort valley positions
    valleys = sorted(peaks.tolist())

    # Determine the text region boundaries
    # First, find the overall text extent (where projection > small threshold)
    text_threshold = max_proj * 0.02  # 2% of max
    text_rows = np.where(projection > text_threshold)[0]

    if len(text_rows) == 0:
        return []

    text_start = text_rows[0]
    text_end = text_rows[-1]

    # Build line boundaries: each line is between consecutive valleys
    # Include edges: text_start to first valley, and last valley to text_end
    split_points = [text_start] + valleys + [text_end]
    boundaries = []

    for i in range(len(split_points) - 1):
        y_start = split_points[i]
        y_end = split_points[i + 1]

        # Check if this region actually contains significant ink
        region_proj = projection[y_start:y_end]
        if np.max(region_proj) > max_proj * 0.05:  # At least 5% of max
            boundaries.append((y_start, y_end))

    return boundaries


def find_line_boundaries_threshold(projection: np.ndarray,
                                   threshold_ratio: float = VALLEY_THRESHOLD) -> list[tuple[int, int]]:
    """
    Original threshold-based approach: find text line boundaries from the projection profile.

    Strategy:
        1. Compute a threshold = threshold_ratio × max(projection)
        2. Regions above the threshold are text lines
        3. Regions below are gaps between lines
    """
    max_val = np.max(projection)
    if max_val == 0:
        return []

    threshold = threshold_ratio * max_val
    is_text = projection > threshold

    # Find transitions: 0→1 = line start, 1→0 = line end
    diff = np.diff(is_text.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases: text starts at row 0 or extends to last row
    if is_text[0]:
        starts = np.insert(starts, 0, 0)
    if is_text[-1]:
        ends = np.append(ends, len(projection))

    # Pair up starts and ends
    boundaries = list(zip(starts.tolist(), ends.tolist()))
    return boundaries


def merge_close_lines(boundaries: list[tuple[int, int]],
                      min_gap: int = 5) -> list[tuple[int, int]]:
    """
    Merge line regions that are separated by a very small gap.
    This handles cases where diacritics above a line create a false valley.
    """
    if not boundaries:
        return []

    merged = [boundaries[0]]
    for start, end in boundaries[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < min_gap:
            # Merge with previous line
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def filter_lines(boundaries: list[tuple[int, int]],
                 min_h: int = MIN_LINE_HEIGHT,
                 max_h: int = MAX_LINE_HEIGHT) -> list[tuple[int, int]]:
    """
    Remove detected regions that are too short (noise) or too tall
    (likely multiple merged lines or non-text regions).
    """
    return [(s, e) for s, e in boundaries if min_h <= (e - s) <= max_h]


def find_horizontal_extent(binary_line: np.ndarray, padding: int = LINE_PADDING_X) -> tuple[int, int]:
    """
    Find the leftmost and rightmost ink pixels in a line crop.
    Trims empty whitespace on the sides for tighter crops.
    """
    ink = (binary_line < 128).astype(np.uint8)
    col_sums = np.sum(ink, axis=0)
    nonzero = np.where(col_sums > 0)[0]

    if len(nonzero) == 0:
        return 0, binary_line.shape[1]

    x_start = max(0, nonzero[0] - padding)
    x_end = min(binary_line.shape[1], nonzero[-1] + padding)
    return x_start, x_end


def segment_page(binary_image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Full line segmentation pipeline for a single page.

    Returns:
        List of (y_start, y_end, x_start, x_end) bounding boxes
        for each detected text line.
    """
    projection = compute_horizontal_projection(binary_image)
    smoothed = smooth_projection(projection)

    # Try local-minima approach first (better for dense text)
    boundaries = find_line_boundaries_minima(smoothed)
    used_minima = len(boundaries) >= 3

    # If minima approach found too few lines, try threshold
    if not used_minima:
        threshold_boundaries = find_line_boundaries_threshold(smoothed)
        if len(threshold_boundaries) > len(boundaries):
            boundaries = threshold_boundaries

    # Only merge close lines for threshold approach.
    # Minima approach produces contiguous boundaries (end == next start)
    # so merging would collapse everything into one region.
    if not used_minima:
        boundaries = merge_close_lines(boundaries)

    boundaries = filter_lines(boundaries)

    line_boxes = []
    for y_start, y_end in boundaries:
        line_crop = binary_image[y_start:y_end, :]
        x_start, x_end = find_horizontal_extent(line_crop)
        line_boxes.append((y_start, y_end, x_start, x_end))

    return line_boxes


def crop_lines(image: np.ndarray, line_boxes: list[tuple[int, int, int, int]],
               pad_y: int = LINE_PADDING_Y, pad_x: int = LINE_PADDING_X) -> list[np.ndarray]:
    """
    Crop line regions from the image with padding.
    Uses the ORIGINAL (non-binarized) image for the actual crops,
    because TrOCR performs better on grayscale than on hard binary.
    """
    h, w = image.shape[:2]
    crops = []

    for y_start, y_end, x_start, x_end in line_boxes:
        y0 = max(0, y_start - pad_y)
        y1 = min(h, y_end + pad_y)
        x0 = max(0, x_start - pad_x)
        x1 = min(w, x_end + pad_x)

        crop = image[y0:y1, x0:x1]
        if crop.size > 0:
            crops.append(crop)

    return crops


def save_debug_overlay(image: np.ndarray, line_boxes: list[tuple[int, int, int, int]],
                       output_path: Path) -> None:
    """
    Save a visualization with detected line bounding boxes overlaid on the page.
    Useful for inspecting and tuning segmentation parameters.
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for i, (y_start, y_end, x_start, x_end) in enumerate(line_boxes):
        color = (0, 255, 0)  # Green boxes
        cv2.rectangle(vis, (x_start, y_start), (x_end, y_end), color, 2)
        cv2.putText(vis, str(i + 1), (x_start, y_start - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(str(output_path), vis)


def segment_and_save(preprocessed_path: Path,
                     lines_dir: Path = LINES_DIR,
                     debug_dir: Path | None = DEBUG_DIR,
                     save_debug: bool = True,
                     method: str | None = None) -> list[Path]:
    """
    Segment a single preprocessed page into lines and save them.
    Supports both 'kraken' (default, baseline layout analysis) and 'projection' (legacy).

    Returns:
        List of paths to saved line crop images.
    """
    if method is None:
        from config import SEGMENTATION_METHOD
        method = SEGMENTATION_METHOD

    color_img = cv2.imread(str(preprocessed_path), cv2.IMREAD_COLOR)
    if color_img is None:
        raise ValueError(f"Could not load: {preprocessed_path}")

    # Convert to grayscale for baseline analysis and mask calculations
    if len(color_img.shape) == 3 and color_img.shape[2] == 3:
        binary = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    else:
        binary = color_img

    page_stem = preprocessed_path.stem
    page_lines_dir = lines_dir / page_stem
    page_lines_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    if method == "kraken":
        if not HAS_KRAKEN:
            logger.warning("Kraken is requested but not installed. Falling back to projection method.")
            method = "projection"

    if method == "kraken":
        from PIL import Image
        from kraken import blla

        im = Image.open(preprocessed_path)

        # Run BLLA segmenter
        try:
            baseline_seg = blla.segment(im, model=KRAKEN_MODEL)
        except Exception as e:
            logger.error("Kraken segmentation failed on %s: %s", preprocessed_path.name, str(e))
            # Fall back to projection if kraken fails on this page
            logger.info("Falling back to projection method on %s", preprocessed_path.name)
            method = "projection"

        if method == "kraken":
            lines = getattr(baseline_seg, 'lines', []) if not isinstance(baseline_seg, dict) else baseline_seg.get('lines', [])

            if not lines:
                logger.warning("No lines detected by Kraken in %s", preprocessed_path.name)
                return []

            # Perform cropping using polygon masking
            for i, line in enumerate(lines):
                boundary = getattr(line, 'boundary', None) if not isinstance(line, dict) else line.get('boundary', None)
                if not boundary:
                    continue

                boundary_pts = np.array(boundary, dtype=np.int32)

                # 1. Create a mask of the polygon
                mask = np.zeros(binary.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [boundary_pts], 255)

                # 2. Mask background: anything outside the polygon is set to white (255, 255, 255)
                masked_img = color_img.copy()
                masked_img[mask == 0] = (255, 255, 255)

                # 3. Crop to the bounding box of the polygon
                x0, y0, w, h = cv2.boundingRect(boundary_pts)

                if w <= 0 or h <= 0:
                    continue

                from config import LINE_PADDING_Y, LINE_PADDING_X
                H, W = binary.shape[:2]
                y_start = max(0, y0 - LINE_PADDING_Y)
                y_end = min(H, y0 + h + LINE_PADDING_Y)
                x_start = max(0, x0 - LINE_PADDING_X)
                x_end = min(W, x0 + w + LINE_PADDING_X)

                crop = masked_img[y_start:y_end, x_start:x_end]
                if crop.size > 0:
                    out_path = page_lines_dir / f"line_{i + 1:03d}.png"
                    cv2.imwrite(str(out_path), crop)
                    saved_paths.append(out_path)

            # Save debug overlays
            if save_debug and debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"{page_stem}_lines.png"
                save_debug_overlay_kraken(binary, lines, debug_path)

            logger.debug("%s → %d lines (kraken)", preprocessed_path.name, len(saved_paths))
            return saved_paths

    # Legacy projection fallback or explicit choice
    if method == "projection" or method != "kraken":
        line_boxes = segment_page(binary)

        if not line_boxes:
            logger.warning("No lines detected in %s", preprocessed_path.name)
            return []

        # Crop from the color image
        crops = crop_lines(color_img, line_boxes)

        for i, crop in enumerate(crops):
            out_path = page_lines_dir / f"line_{i + 1:03d}.png"
            cv2.imwrite(str(out_path), crop)
            saved_paths.append(out_path)

        if save_debug and debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"{page_stem}_lines.png"
            save_debug_overlay(binary, line_boxes, debug_path)

        logger.debug("%s → %d lines (projection)", preprocessed_path.name, len(saved_paths))
        return saved_paths


def segment_all(input_dir: Path = PREPROCESSED_DIR,
                output_dir: Path = LINES_DIR,
                save_debug: bool = True,
                method: str | None = None) -> dict[str, list[Path]]:
    """
    Segment all preprocessed pages into lines.

    Returns:
        Dict mapping page stem → list of line image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    page_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    logger.info("Segmenting %d pages into lines …", len(page_files))
    results: dict[str, list[Path]] = {}
    total_lines = 0

    for page_path in tqdm(page_files, desc="Line segmentation", unit="page"):
        try:
            line_paths = segment_and_save(page_path, output_dir, save_debug=save_debug, method=method)
            results[page_path.stem] = line_paths
            total_lines += len(line_paths)
        except Exception:
            logger.exception("Failed to segment %s", page_path.name)

    logger.info("Total lines extracted: %d from %d pages", total_lines, len(page_files))
    return results
