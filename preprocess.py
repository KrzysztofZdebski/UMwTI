"""
Stage 1B: Image preprocessing — deskew, grayscale, binarization, margin crop.
Optimized for historical handwritten documents on aged paper.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    MARGIN_CROP_FRACTION,
    PAGES_DIR,
    PREPROCESSED_DIR,
    SAUVOLA_K,
    SAUVOLA_WINDOW_SIZE,
    BINARIZE_ON_PREPROCESS,
)

logger = logging.getLogger(__name__)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale. No-op if already single-channel."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def crop_margins(image: np.ndarray, fraction: float = MARGIN_CROP_FRACTION) -> np.ndarray:
    """
    Remove a fixed fraction of pixels from each edge.
    Eliminates dark scanner borders and binding shadows.
    """
    h, w = image.shape[:2]
    dy = int(h * fraction)
    dx = int(w * fraction)
    return image[dy : h - dy, dx : w - dx]


def estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate document skew angle using Hough line transform.
    Returns angle in degrees (positive = counter-clockwise tilt).
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,  # At least 1/4 of page width
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        return 0.0

    # Compute angles of all detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (text lines)
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def deskew(image: np.ndarray, angle: float | None = None) -> np.ndarray:
    """
    Rotate image to correct skew. If angle is None, auto-estimate it.
    Uses white background fill for rotated corners.
    """
    if angle is None:
        gray = to_grayscale(image)
        angle = estimate_skew_angle(gray)

    if abs(angle) < 0.3:
        # Skew is negligible; skip rotation to avoid interpolation artifacts
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Use white border for text documents
    border_value = 255 if len(image.shape) == 2 else (255, 255, 255)
    rotated = cv2.warpAffine(
        image, rot_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    logger.debug("Deskewed by %.2f°", angle)
    return rotated


def binarize_sauvola(gray: np.ndarray, window_size: int = SAUVOLA_WINDOW_SIZE,
                     k: float = SAUVOLA_K) -> np.ndarray:
    """
    Sauvola adaptive binarization — handles uneven lighting from paper aging,
    fold creases, and ink bleed far better than global Otsu.

    Returns a binary image: 0 = ink, 255 = background.
    """
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    # Compute local mean and standard deviation
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (window_size, window_size))
    mean_sq = cv2.blur(gray_f ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    # Sauvola threshold: T(x,y) = mean * (1 + k * (std / R - 1))
    R = 128.0  # Dynamic range of standard deviation
    threshold = mean * (1.0 + k * (std / R - 1.0))

    binary = np.where(gray_f > threshold, 255, 0).astype(np.uint8)
    return binary


def denoise(binary: np.ndarray) -> np.ndarray:
    """
    Light morphological denoising: remove small specks without
    damaging thin pen strokes.
    """
    # Small opening to remove isolated noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cleaned


def preprocess_single(image_path: Path, output_dir: Path) -> Path:
    """
    Full preprocessing pipeline for a single page image:
      1. Load → 2. Crop margins → 3. Deskew color image → 4. Optional binarization & denoising

    Returns path to the preprocessed output image.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # 1. Crop scanner margins
    img = crop_margins(img)

    # 2. Deskew the color image (internally estimates angle on grayscale)
    img = deskew(img)

    # 3. Optional binarization & denoising
    if BINARIZE_ON_PREPROCESS:
        gray = to_grayscale(img)
        binary = binarize_sauvola(gray)
        img = denoise(binary)

    # Save (will be either full color deskewed or B&W binarized)
    out_path = output_dir / image_path.name
    cv2.imwrite(str(out_path), img)

    return out_path


def preprocess_all(input_dir: Path = PAGES_DIR,
                   output_dir: Path = PREPROCESSED_DIR) -> list[Path]:
    """
    Preprocess all page images in the input directory.
    Returns list of preprocessed image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    page_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )

    logger.info("Preprocessing %d page images …", len(page_files))
    results: list[Path] = []

    for page_path in tqdm(page_files, desc="Preprocessing", unit="page"):
        try:
            out = preprocess_single(page_path, output_dir)
            results.append(out)
        except Exception:
            logger.exception("Failed to preprocess %s", page_path.name)

    logger.info("Preprocessed %d / %d pages successfully", len(results), len(page_files))
    return results
