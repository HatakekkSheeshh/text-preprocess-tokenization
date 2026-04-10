from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "outputs" / "load_text_samples"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.load_data import load_text_dataset

DATASET_NAME = ("one-billion-word", "wikitext-103", "text8", "enwik8")


def validate_dataset_name(dataset_name: str) -> None:
    if dataset_name not in DATASET_NAME:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Expected one of: {', '.join(DATASET_NAME)}"
        )

def write_split_samples(
    *,
    dataset_name: str,
    split_name: str,
    output_dir: Path,
    max_texts: int,
    max_chars_per_text: int | None,
) -> int:
    validate_dataset_name(dataset_name)
    text_dataset = load_text_dataset(dataset_name)
    split_output_dir = output_dir / dataset_name / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    written_count = 0
    for index, text in enumerate(text_dataset.iter_texts(split_name), start=1):
        if max_chars_per_text is not None:
            text = text[:max_chars_per_text]

        output_path = split_output_dir / f"text_{index:03d}.txt"
        output_path.write_text(text, encoding="utf-8")
        written_count += 1

        if written_count >= max_texts:
            break

    return written_count


def export_text_samples(
    dataset_name: str,
    *,
    output_dir: Path,
    max_texts_per_split: int,
    max_chars_per_text: int | None,
) -> dict[str, int]:
    validate_dataset_name(dataset_name)
    text_dataset = load_text_dataset(dataset_name)
    results: dict[str, int] = {}

    for split_name in text_dataset.split_names:
        results[split_name] = write_split_samples(
            dataset_name=dataset_name,
            split_name=split_name,
            output_dir=output_dir,
            max_texts=max_texts_per_split,
            max_chars_per_text=max_chars_per_text,
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sample text files from load_text_dataset().")
    parser.add_argument("--dataset", type=str, default="enwik8")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-texts-per-split", type=int, default=3)
    parser.add_argument("--max-chars-per-text", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = export_text_samples(
        args.dataset,
        output_dir=args.output_dir,
        max_texts_per_split=args.max_texts_per_split,
        max_chars_per_text=args.max_chars_per_text,
    )

    print(f"Saved sample texts for dataset '{args.dataset}' to: {args.output_dir}")
    for split_name, count in results.items():
        print(f"- {split_name}: {count} file(s)")


if __name__ == "__main__":
    main()
