"""
Central configuration for the letter OCR preprocessing pipeline.
All paths and tunable parameters live here.
"""

from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────────
INPUT_DIR = Path(r"c:\Users\KMZde\Desktop\AGH\UMwTI\Listy")
OUTPUT_DIR = Path(r"c:\Users\KMZde\Desktop\AGH\UMwTI\output")

# Sub-directories created automatically
PAGES_DIR = OUTPUT_DIR / "1_pages"           # Extracted full-page images
PREPROCESSED_DIR = OUTPUT_DIR / "2_preprocessed"  # Deskewed & binarized pages
LINES_DIR = OUTPUT_DIR / "3_lines"           # Individual line crops
TRANSCRIPTIONS_DIR = OUTPUT_DIR / "4_transcriptions"  # Text transcription outputs
DEBUG_DIR = OUTPUT_DIR / "debug"             # Visualizations (segmentation overlays)

# ─── PDF Extraction ─────────────────────────────────────────────────────────────
PDF_DPI = 300           # Resolution for rasterizing PDF pages (300-400 recommended)

# ─── Preprocessing ──────────────────────────────────────────────────────────────
TARGET_DPI = 300        # If images are higher res, downscale to this
SAUVOLA_WINDOW_SIZE = 51   # Odd integer; window for adaptive binarization
SAUVOLA_K = 0.2            # Sensitivity: lower = more aggressive binarization
BINARIZE_ON_PREPROCESS = False  # Set to True to output B&W binarized pages (default: False, keeps original colors)

# ─── Line Segmentation ─────────────────────────────────────────────────────────
SEGMENTATION_METHOD = "kraken"  # Default segmentation method ("kraken" or "projection")
KRAKEN_MODEL = None            # Path to custom BLLA model, or None for default
MIN_LINE_HEIGHT = 20     # px — ignore "lines" shorter than this (noise)
MAX_LINE_HEIGHT = 300    # px — ignore regions taller than this (likely not a single line)
LINE_PADDING_Y = 10      # px — vertical padding above/below each line crop
LINE_PADDING_X = 5       # px — horizontal padding left/right of each line crop
SMOOTHING_KERNEL = 25    # Kernel size for smoothing the horizontal projection profile
VALLEY_THRESHOLD = 0.15  # Fraction of max projection value below which we call it a gap
MARGIN_CROP_FRACTION = 0.03  # Crop this fraction off each edge to remove dark borders/margins

# ─── TrOCR Transcription ──────────────────────────────────────────────────────
TROCR_MODEL_BASE = "microsoft/trocr-base-handwritten"    # Default English handwriting model
POLISH_ENCODER = "google/vit-base-patch16-384"           # Standard ViT for image processing patch
POLISH_DECODER = "allegro/herbert-base-cased"           # Polish-trained tokenizer/decoder (HerBERT)
TROCR_FINETUNED_SAMPLE300_DIR = OUTPUT_DIR / "models" / "trocr_sample300"
TRANSCRIPTION_BATCH_SIZE = 8  # Lines per batch during inference (lower if running out of memory)
