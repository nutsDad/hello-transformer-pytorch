"""Create reproducible train/dev/test splits for decoder-only text datasets."""

import argparse
import json
import random
from pathlib import Path

from tools.data_loader import DecoderOnlyDataset, clean_decoder_only_texts


def split_items(items, dev_ratio, test_ratio, seed):
    """Shuffle items deterministically and return non-overlapping dataset splits."""
    if not 0 <= dev_ratio < 1 or not 0 <= test_ratio < 1 or dev_ratio + test_ratio >= 1:
        raise ValueError("dev_ratio and test_ratio must be in [0, 1) and sum to less than 1")
    if len(items) < 3:
        raise ValueError("At least three cleaned documents are required to make all splits.")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, round(len(shuffled) * test_ratio))
    dev_size = max(1, round(len(shuffled) * dev_ratio))
    train_size = len(shuffled) - dev_size - test_size
    if train_size < 1:
        raise ValueError("Split ratios leave no training documents.")
    return (
        shuffled[:train_size],
        shuffled[train_size : train_size + dev_size],
        shuffled[train_size + dev_size :],
    )


def write_json(path, items):
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def run(input_path, output_dir, dev_ratio=0.05, test_ratio=0.05, seed=42, min_characters=2):
    """Clean input documents and write train.json, dev.json, test.json, and a report."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    raw_items = DecoderOnlyDataset._load_items(input_path)
    cleaned, statistics = clean_decoder_only_texts(raw_items, min_characters=min_characters)
    train_items, dev_items, test_items = split_items(cleaned, dev_ratio, test_ratio, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train.json", train_items)
    write_json(output_dir / "dev.json", dev_items)
    write_json(output_dir / "test.json", test_items)
    report = {
        "source": str(input_path),
        "seed": seed,
        "dev_ratio": dev_ratio,
        "test_ratio": test_ratio,
        "statistics": statistics,
        "splits": {
            "train_documents": len(train_items),
            "dev_documents": len(dev_items),
            "test_documents": len(test_items),
        },
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UTF-8 .txt or decoder-only .json input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-characters", type=int, default=2)
    arguments = parser.parse_args()
    result = run(
        arguments.input,
        arguments.output_dir,
        arguments.dev_ratio,
        arguments.test_ratio,
        arguments.seed,
        arguments.min_characters,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
