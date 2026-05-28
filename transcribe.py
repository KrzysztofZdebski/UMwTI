"""
Stage 4: Transcribe line crops using TrOCR.

Supports two model variants:
  - Base:   microsoft/trocr-base-handwritten  (default, faster, ~350 M params)
  - Polish: Custom blank-slate TrOCR using a standard ViT encoder (google/vit-base-patch16-384)
            and a Polish HerBERT decoder (allegro/herbert-base-cased) for future fine-tuning.

Usage (standalone):
    python transcribe.py                     # Base model, all pages
    python transcribe.py --polish            # Polish custom model
    python transcribe.py --sample 5          # Only first 5 pages
"""

import csv
import logging
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from config import (
    LINES_DIR,
    TRANSCRIPTIONS_DIR,
    TRANSCRIPTION_BATCH_SIZE,
    TROCR_MODEL_BASE,
    POLISH_ENCODER,
    POLISH_DECODER,
    TROCR_FINETUNED_SAMPLE300_DIR,
)

logger = logging.getLogger(__name__)

# Re-use the dataset metadata path from build_dataset
DATASET_DIR = Path(__file__).resolve().parent / "output" / "dataset"
METADATA_FILE = DATASET_DIR / "metadata.csv"


# ─── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_variant: str = "base", finetuned_model_dir: Path | None = None):
    """
    Load the TrOCR processor and model.

    Args:
        use_polish: If True, assemble a custom Polish blank-slate model.
                    Otherwise, load the default English base model.

    Returns:
        Tuple of (processor, model, device_string).
    """
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if model_variant == "polish":
        logger.info("Assembling custom Polish TrOCR model:")
        logger.info("  - Encoder: %s", POLISH_ENCODER)
        logger.info("  - Decoder: %s", POLISH_DECODER)

        feature_extractor = ViTImageProcessor.from_pretrained(POLISH_ENCODER)
        tokenizer = AutoTokenizer.from_pretrained(POLISH_DECODER)

        # Combine into blank-slate TrOCR architecture
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            POLISH_ENCODER,
            POLISH_DECODER
        )
        # Set special tokens and vocab size appropriately before starting
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        if model.generation_config is not None:
            model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.eos_token_id = tokenizer.sep_token_id

        # Wrap in a standard TrOCRProcessor
        processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
    elif model_variant == "finetuned":
        model_path = finetuned_model_dir or TROCR_FINETUNED_SAMPLE300_DIR
        logger.info("Loading fine-tuned TrOCR model from: %s", model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Fine-tuned model directory not found: {model_path}. "
                "Train it first with train_trocr_sample300.py."
            )
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
    else:
        logger.info("Loading TrOCR base model: %s", TROCR_MODEL_BASE)
        processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_BASE)
        model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_BASE)

    model.to(device)
    model.eval()

    logger.info("Model loaded on device: %s", device)
    return processor, model, device


# ─── Single-line transcription ──────────────────────────────────────────────────

def transcribe_lines_batch(images: list[Image.Image], processor, model, device: str) -> list[str]:
    """
    Transcribe a batch of PIL Images through TrOCR.

    Args:
        images:    List of PIL RGB images (one per text line).
        processor: TrOCR processor instance.
        model:     TrOCR model instance.
        device:    'cuda' or 'cpu'.

    Returns:
        List of decoded text strings, one per input image.
    """
    import torch

    # Processor expects RGB images
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=256)

    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return texts


# ─── Per-page transcription ────────────────────────────────────────────────────

def transcribe_page(page_dir: Path, processor, model, device: str,
                    batch_size: int = TRANSCRIPTION_BATCH_SIZE) -> list[tuple[int, str]]:
    """
    Transcribe all line images in a single page directory.

    Args:
        page_dir:   Path to a directory containing line_001.png, line_002.png, etc.
        processor:  TrOCR processor.
        model:      TrOCR model.
        device:     Device string.
        batch_size: Number of lines to process in one forward pass.

    Returns:
        Sorted list of (line_number, transcribed_text) tuples.
    """
    line_files = sorted(page_dir.glob("line_*.png"))
    if not line_files:
        logger.warning("No line images found in %s", page_dir)
        return []

    results = []

    # Process in batches
    for i in range(0, len(line_files), batch_size):
        batch_files = line_files[i : i + batch_size]
        images = []
        line_nums = []

        for lf in batch_files:
            img = Image.open(lf).convert("RGB")
            images.append(img)
            line_num = int(lf.stem.split("_")[1])
            line_nums.append(line_num)

        texts = transcribe_lines_batch(images, processor, model, device)

        for line_num, text in zip(line_nums, texts):
            results.append((line_num, text.strip()))

    # Sort by line number
    results.sort(key=lambda x: x[0])
    return results


# ─── Full transcription pipeline ───────────────────────────────────────────────

def transcribe_all(model_variant: str = "base",
                   sample: int | None = None,
                   lines_dir: Path = LINES_DIR,
                   output_dir: Path = TRANSCRIPTIONS_DIR,
                   finetuned_model_dir: Path | None = None) -> dict[str, list[tuple[int, str]]]:
    """
    Transcribe all (or sampled) pages and save results.

    Args:
        model_variant: "base", "polish", or "finetuned".
        sample:     If set, only process the first N page directories.
        lines_dir:  Directory containing page sub-directories with line crops.
        output_dir: Where to write per-page .txt transcription files.
        finetuned_model_dir: Optional directory override for the fine-tuned model.

    Returns:
        Dict mapping page_name -> list of (line_number, text).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather page directories
    page_dirs = sorted([d for d in lines_dir.iterdir() if d.is_dir()])
    if not page_dirs:
        logger.error("No page directories found in %s", lines_dir)
        return {}

    if sample:
        page_dirs = page_dirs[:sample]
        logger.info("Sampling: processing %d of %d pages", len(page_dirs),
                     len(list(lines_dir.iterdir())))

    # Load model once
    processor, model, device = load_model(
        model_variant=model_variant,
        finetuned_model_dir=finetuned_model_dir,
    )

    all_results = {}

    for page_dir in tqdm(page_dirs, desc="Transcribing pages", unit="page"):
        page_name = page_dir.name
        results = transcribe_page(page_dir, processor, model, device)
        all_results[page_name] = results

        # Write per-page text file
        txt_path = output_dir / f"{page_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for line_num, text in results:
                f.write(f"{text}\n")

        logger.debug("Page %s: %d lines transcribed", page_name, len(results))

    # Update metadata.csv with predictions
    _update_metadata(all_results)

    return all_results


def _update_metadata(results: dict[str, list[tuple[int, str]]]) -> None:
    """
    Update the metadata.csv file with transcribed text.

    Reads the existing CSV, fills in the 'text' column for pages that were
    transcribed, and writes it back.
    """
    if not METADATA_FILE.exists():
        logger.warning("metadata.csv not found at %s — skipping update", METADATA_FILE)
        return

    # Read existing rows
    rows = []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Build a lookup: (page, line_number) -> text
    lookup = {}
    for page_name, lines in results.items():
        for line_num, text in lines:
            lookup[(page_name, line_num)] = text

    # Update rows
    updated_count = 0
    for row in rows:
        key = (row["page"], int(row["line_number"]))
        if key in lookup:
            row["text"] = lookup[key]
            updated_count += 1

    # Write back
    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text", "page", "line_number"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Updated %d entries in metadata.csv", updated_count)


# ─── Standalone entry point ────────────────────────────────────────────────────

def main():
    import argparse
    import sys
    import os

    # Fix Windows console encoding
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Stage 4: Transcribe line crops using TrOCR"
    )
    parser.add_argument(
        "--model", type=str, choices=["base", "polish", "finetuned"], default="base",
        help="Model variant to use for transcription."
    )
    parser.add_argument(
        "--polish", action="store_true",
        help="Deprecated alias for --model polish."
    )
    parser.add_argument(
        "--finetuned-model-dir", type=str, default=None,
        help="Path to a fine-tuned model directory (used only with --model finetuned)."
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Process only the first N pages (for quick testing)."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    t = time.time()
    model_variant = "polish" if args.polish else args.model
    finetuned_dir = Path(args.finetuned_model_dir) if args.finetuned_model_dir else None
    results = transcribe_all(
        model_variant=model_variant,
        sample=args.sample,
        finetuned_model_dir=finetuned_dir,
    )

    total_lines = sum(len(v) for v in results.values())
    print(f"\nTranscribed {total_lines} lines from {len(results)} pages "
          f"in {time.time() - t:.1f}s")
    print(f"Output: {TRANSCRIPTIONS_DIR}")

    # Print a sample of transcriptions
    if results:
        first_page = next(iter(results))
        lines = results[first_page]
        print(f"\n--- Sample from {first_page} (first 5 lines) ---")
        for line_num, text in lines[:5]:
            print(f"  Line {line_num:03d}: {text}")


if __name__ == "__main__":
    main()
