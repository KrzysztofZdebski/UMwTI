"""
Fine-tune TrOCR on output/sample_300 and evaluate on a held-out split.

Expected inputs:
  - output/sample_300/                  (image files)
  - output/sample_300_transcribed.txt   (tab-separated: image_name<TAB>text)

Usage:
  python finetune_trocr_sample300.py
  python finetune_trocr_sample300.py --epochs 12 --test-size 0.2
"""

from __future__ import annotations

import argparse
import inspect
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from config import OUTPUT_DIR, TROCR_MODEL_BASE, TROCR_FINETUNED_SAMPLE300_DIR

logger = logging.getLogger(__name__)

SAMPLE_IMAGES_DIR = OUTPUT_DIR / "sample_300"
SAMPLE_TRANSCRIPTIONS_FILE = OUTPUT_DIR / "sample_300_transcribed.txt"


def _build_training_args(output_dir: Path, args, use_fp16: bool) -> Seq2SeqTrainingArguments:
    """
    Build Seq2SeqTrainingArguments across transformers API variants.

    Some versions accept `evaluation_strategy`, while others use `eval_strategy`.
    """
    common_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "predict_with_generate": True,
        "fp16": use_fp16,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "report_to": "none",
    }

    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "evaluation_strategy" in params:
        common_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in params:
        common_kwargs["eval_strategy"] = "epoch"
    else:
        logger.warning("No eval strategy arg found; falling back to no periodic evaluation.")

    return Seq2SeqTrainingArguments(**common_kwargs)


def _build_trainer(
    model: VisionEncoderDecoderModel,
    training_args: Seq2SeqTrainingArguments,
    train_ds: Dataset,
    test_ds: Dataset,
    processor: TrOCRProcessor,
) -> Seq2SeqTrainer:
    """Build Seq2SeqTrainer across transformers API variants."""
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": test_ds,
        "data_collator": TrOCRDataCollator(),
        "compute_metrics": build_compute_metrics(processor),
    }

    params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in params:
        kwargs["tokenizer"] = processor.tokenizer
    elif "processing_class" in params:
        kwargs["processing_class"] = processor
    else:
        logger.warning("No tokenizer/processing_class argument found in Seq2SeqTrainer.")

    return Seq2SeqTrainer(**kwargs)


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance with dynamic programming."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                curr[j - 1] + 1,      # insertion
                prev[j] + 1,          # deletion
                prev[j - 1] + cost,   # substitution
            ))
        prev = curr
    return prev[-1]


def _character_error_rate(refs: list[str], preds: list[str]) -> float:
    total_dist = 0
    total_chars = 0
    for ref, pred in zip(refs, preds):
        total_dist += _levenshtein_distance(ref, pred)
        total_chars += max(1, len(ref))
    return total_dist / total_chars


def load_samples(images_dir: Path, transcription_file: Path) -> list[dict[str, str]]:
    """Load labeled samples and validate image existence."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not transcription_file.exists():
        raise FileNotFoundError(f"Transcription file not found: {transcription_file}")

    records: list[dict[str, str]] = []
    with open(transcription_file, "r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"Line {idx} in {transcription_file} is not tab-separated.")

            image_name, text = line.split("\t", 1)
            image_path = images_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for line {idx}: {image_name}")

            records.append({"image_path": str(image_path), "text": text})

    if not records:
        raise ValueError("No valid samples were loaded.")

    logger.info("Loaded %d labeled samples.", len(records))
    return records


def build_datasets(records: list[dict[str, str]], test_size: float, seed: int) -> tuple[Dataset, Dataset]:
    """Create train/test HF datasets."""
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    train_ds = split["train"]
    test_ds = split["test"]
    logger.info("Split dataset: %d train / %d test", len(train_ds), len(test_ds))
    return train_ds, test_ds


def preprocess_dataset(ds: Dataset, processor: TrOCRProcessor, max_target_length: int) -> Dataset:
    """Tokenize images and text for TrOCR training."""

    def _map_fn(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values

        labels = processor.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=max_target_length,
            truncation=True,
        ).input_ids

        # Ignore pad tokens in the loss
        labels = [
            [token if token != processor.tokenizer.pad_token_id else -100 for token in seq]
            for seq in labels
        ]

        return {
            "pixel_values": pixel_values.tolist(),
            "labels": labels,
        }

    return ds.map(
        _map_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc="Preprocessing",
    )


@dataclass
class TrOCRDataCollator:
    """Batch collator for preprocessed TrOCR tensors."""

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        pixel_values = torch.tensor([f["pixel_values"] for f in features], dtype=torch.float32)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}


def build_compute_metrics(processor: TrOCRProcessor):
    """Build CER + exact-match metrics for Trainer."""

    def _compute_metrics(eval_preds):
        pred_ids, label_ids = eval_preds
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Replace ignore index to decode labels
        labels = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_texts = processor.batch_decode(labels, skip_special_tokens=True)

        pred_texts = [p.strip() for p in pred_texts]
        label_texts = [l.strip() for l in label_texts]

        cer = _character_error_rate(label_texts, pred_texts)
        exact_match = sum(p == l for p, l in zip(pred_texts, label_texts)) / max(1, len(label_texts))
        return {"cer": cer, "exact_match": exact_match}

    return _compute_metrics


def run_qualitative_test(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    test_records: list[dict[str, str]],
    num_examples: int,
) -> None:
    """Print sample predictions from the test split."""
    if not test_records:
        logger.warning("No test samples available for qualitative test.")
        return

    model.eval()
    device = model.device

    logger.info("Sample predictions on held-out set:")
    subset = test_records[: min(num_examples, len(test_records))]
    for i, record in enumerate(subset, start=1):
        image = Image.open(record["image_path"]).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=128)
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        truth = record["text"].strip()
        logger.info("  [%d] GT: %s", i, truth)
        logger.info("      PR: %s", pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on 300 labeled samples and test it.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Train/eval batch size per device.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Fraction for held-out test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-target-length", type=int, default=128, help="Max label token length.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(TROCR_FINETUNED_SAMPLE300_DIR),
        help="Where to save the fine-tuned model.",
    )
    parser.add_argument(
        "--print-test-examples",
        type=int,
        default=10,
        help="How many held-out examples to print after testing.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_samples(SAMPLE_IMAGES_DIR, SAMPLE_TRANSCRIPTIONS_FILE)
    train_raw, test_raw = build_datasets(records, test_size=args.test_size, seed=args.seed)

    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_BASE)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_BASE)

    # Ensure generation tokens are consistently configured.
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if model.generation_config is not None:
        model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.eos_token_id = processor.tokenizer.sep_token_id

    train_ds = preprocess_dataset(train_raw, processor, max_target_length=args.max_target_length)
    test_ds = preprocess_dataset(test_raw, processor, max_target_length=args.max_target_length)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available()
    training_args = _build_training_args(output_dir=output_dir, args=args, use_fp16=use_fp16)

    trainer = _build_trainer(
        model=model,
        training_args=training_args,
        train_ds=train_ds,
        test_ds=test_ds,
        processor=processor,
    )

    logger.info("Starting fine-tuning.")
    trainer.train()

    logger.info("Running final evaluation on held-out split.")
    eval_metrics = trainer.evaluate()
    logger.info("Final metrics: %s", eval_metrics)

    logger.info("Saving fine-tuned model to %s", output_dir)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # Use raw test records for readable qualitative examples.
    test_records = [test_raw[i] for i in range(len(test_raw))]
    run_qualitative_test(
        model=trainer.model,
        processor=processor,
        test_records=test_records,
        num_examples=args.print_test_examples,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
