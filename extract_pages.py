"""
Stage 1A: Extract images from source files (PDF pages → PNG, copy JPGs).
Handles both multi-page PDFs and standalone image files.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

from config import INPUT_DIR, PAGES_DIR, PDF_DPI

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def extract_pages_from_pdf(pdf_path: Path, output_dir: Path, dpi: int = PDF_DPI) -> list[Path]:
    """
    Render each page of a PDF as a PNG image at the specified DPI.

    Returns:
        List of paths to the extracted page images.
    """
    doc = fitz.open(pdf_path)
    stem = pdf_path.stem
    extracted = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        # Compute zoom factor: default PDF resolution is 72 DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        out_name = f"{stem}_p{page_idx + 1:03d}.png"
        out_path = output_dir / out_name
        pix.save(str(out_path))
        extracted.append(out_path)

    doc.close()
    return extracted


def copy_image(image_path: Path, output_dir: Path) -> Path:
    """
    Copy a standalone image file to the output directory as PNG.
    Normalizes format to PNG for consistency downstream.
    """
    stem = image_path.stem
    out_path = output_dir / f"{stem}.png"

    img = Image.open(image_path)
    img.save(str(out_path), format="PNG")
    img.close()

    return out_path


def extract_all(input_dir: Path = INPUT_DIR, output_dir: Path = PAGES_DIR) -> list[Path]:
    """
    Walk the input directory, extract pages from every PDF, and copy
    standalone images. Returns list of all output page image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all source files
    pdfs = sorted(input_dir.glob("*.pdf"))
    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )

    all_pages: list[Path] = []

    logger.info("Extracting pages from %d PDFs …", len(pdfs))
    for pdf_path in tqdm(pdfs, desc="PDFs → pages", unit="file"):
        try:
            pages = extract_pages_from_pdf(pdf_path, output_dir)
            all_pages.extend(pages)
        except Exception:
            logger.exception("Failed to extract pages from %s", pdf_path.name)

    logger.info("Copying %d standalone images …", len(images))
    for img_path in tqdm(images, desc="Images → pages", unit="file"):
        try:
            out = copy_image(img_path, output_dir)
            all_pages.append(out)
        except Exception:
            logger.exception("Failed to copy image %s", img_path.name)

    logger.info("Total pages extracted: %d", len(all_pages))
    return all_pages
