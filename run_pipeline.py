"""
Main entry point: run the full preprocessing pipeline (Stages 1-4).

Usage:
    python run_pipeline.py                  # Full pipeline
    python run_pipeline.py --stage 1        # Only extract pages from PDFs
    python run_pipeline.py --stage 2        # Only preprocess (deskew + binarize)
    python run_pipeline.py --stage 3        # Only segment lines
    python run_pipeline.py --stage 4        # Only transcribe with TrOCR
    python run_pipeline.py --stage 4 --polish  # Transcribe with custom Polish blank-slate model
    python run_pipeline.py --stage 4 --transcription-model finetuned  # Use fine-tuned model
    python run_pipeline.py --no-debug       # Skip debug visualizations
    python run_pipeline.py --sample 5       # Process only first 5 pages (for testing)
"""

import argparse
import logging
import os
import sys
import time

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import LINES_DIR, OUTPUT_DIR, PAGES_DIR, PREPROCESSED_DIR, TRANSCRIPTIONS_DIR
from extract_pages import extract_all
from preprocess import preprocess_all
from segment_lines import segment_all
from build_dataset import build_metadata, print_dataset_stats
from transcribe import transcribe_all


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with timestamps and module names."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / "pipeline.log", encoding="utf-8"),
        ],
    )


def _banner(text: str) -> None:
    """Print a stage banner using ASCII-safe characters."""
    width = len(text) + 4
    print("\n+" + "=" * width + "+")
    print("|  " + text + "  |")
    print("+" + "=" * width + "+\n")


def run_pipeline(stages: list[int] | None = None,
                 save_debug: bool = True,
                 sample: int | None = None,
                 method: str | None = None,
                 use_polish: bool = False,
                 transcription_model: str = "base") -> None:
    """
    Execute the preprocessing pipeline.

    Args:
        stages: Which stages to run (1, 2, 3, 4). None = all.
        save_debug: Whether to save debug visualizations.
        sample: If set, only process this many pages (for quick testing).
        method: The line segmentation method ('kraken' or 'projection').
        use_polish: If True, use the large TrOCR model for better Polish support.
        transcription_model: TrOCR variant for stage 4 ("base", "polish", "finetuned").
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_all = stages is None
    t_total = time.time()

    # -- Stage 1: Extract pages from PDFs/images --
    if run_all or 1 in stages:
        _banner("STAGE 1: Extracting pages from sources")
        t = time.time()
        pages = extract_all()
        print(f"  [OK] Extracted {len(pages)} pages in {time.time() - t:.1f}s\n")

    # -- Stage 2: Preprocess (deskew + binarize) --
    if run_all or 2 in stages:
        _banner("STAGE 2: Preprocessing pages")

        # If sampling, only process a subset
        if sample:
            from pathlib import Path
            import shutil
            page_files = sorted(PAGES_DIR.iterdir())[:sample]
            sample_dir = OUTPUT_DIR / "_sample_pages"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for p in page_files:
                shutil.copy2(p, sample_dir / p.name)
            t = time.time()
            preprocessed = preprocess_all(input_dir=sample_dir)
        else:
            t = time.time()
            preprocessed = preprocess_all()

        print(f"  [OK] Preprocessed {len(preprocessed)} pages in {time.time() - t:.1f}s\n")

    # -- Stage 3: Segment lines --
    if run_all or 3 in stages:
        _banner("STAGE 3: Segmenting lines")
        t = time.time()
        results = segment_all(save_debug=save_debug, method=method)
        total_lines = sum(len(v) for v in results.values())
        print(f"  [OK] Extracted {total_lines} lines from {len(results)} pages in {time.time() - t:.1f}s\n")

        # Build dataset metadata
        print("  Building dataset metadata ...")
        metadata_path = build_metadata()
        print(f"  [OK] Metadata saved to {metadata_path}\n")

        # Print stats
        print_dataset_stats()

    # -- Stage 4: Transcribe lines with TrOCR --
    if run_all or 4 in stages:
        _banner("STAGE 4: Transcribing lines with TrOCR")
        if use_polish:
            transcription_model = "polish"

        model_labels = {
            "base": "base",
            "polish": "custom Polish blank-slate (ViT+HerBERT)",
            "finetuned": "fine-tuned on sample_300",
        }
        model_label = model_labels.get(transcription_model, transcription_model)
        print(f"  Model: {model_label}")
        if sample:
            print(f"  Sampling: first {sample} pages")
        t = time.time()
        results = transcribe_all(model_variant=transcription_model, sample=sample)
        total_lines = sum(len(v) for v in results.values())
        print(f"  [OK] Transcribed {total_lines} lines from {len(results)} pages in {time.time() - t:.1f}s\n")

        # Print a sample
        if results:
            first_page = next(iter(results))
            lines = results[first_page]
            print(f"  --- Sample from {first_page} (first 3 lines) ---")
            for line_num, text in lines[:3]:
                print(f"    Line {line_num:03d}: {text}")
            print()

    elapsed = time.time() - t_total
    print(f"\n{'-' * 50}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'-' * 50}\n")

    # Print directory structure
    print("Output structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  +-- 1_pages/          <- Full-page images ({_count_files(PAGES_DIR)} files)")
    print(f"  +-- 2_preprocessed/   <- Deskewed & binarized ({_count_files(PREPROCESSED_DIR)} files)")
    print(f"  +-- 3_lines/          <- Line crops ({_count_files(LINES_DIR, recursive=True)} files)")
    print(f"  +-- 4_transcriptions/ <- Text outputs ({_count_files(TRANSCRIPTIONS_DIR)} files)")
    print(f"  +-- dataset/          <- metadata.csv with transcriptions")
    print(f"  +-- debug/            <- Segmentation overlays")


def _count_files(d, recursive=False):
    """Count files in a directory."""
    from pathlib import Path
    d = Path(d)
    if not d.exists():
        return 0
    if recursive:
        return sum(1 for _ in d.rglob("*.png"))
    return sum(1 for _ in d.glob("*") if _.is_file())


def main():
    parser = argparse.ArgumentParser(
        description="OCR Preprocessing Pipeline for Historical Handwritten Letters"
    )
    parser.add_argument(
        "--stage", type=int, nargs="+", choices=[1, 2, 3, 4],
        help="Run only specific stages (1=extract, 2=preprocess, 3=segment, 4=transcribe). Default: all."
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Skip saving debug visualizations (faster)."
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Process only the first N pages (for quick testing)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging."
    )
    parser.add_argument(
        "--method", type=str, choices=["kraken", "projection"], default=None,
        help="Line segmentation method. Options: kraken (default), projection (legacy)."
    )
    parser.add_argument(
        "--polish", action="store_true",
        help="Use custom Polish blank-slate model (ViT + HerBERT) instead of the base model."
    )
    parser.add_argument(
        "--transcription-model",
        type=str,
        choices=["base", "polish", "finetuned"],
        default="base",
        help="Stage 4 model choice. Options: base, polish, finetuned.",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    run_pipeline(
        stages=args.stage,
        save_debug=not args.no_debug,
        sample=args.sample,
        method=args.method,
        use_polish=args.polish,
        transcription_model=args.transcription_model,
    )


if __name__ == "__main__":
    main()
