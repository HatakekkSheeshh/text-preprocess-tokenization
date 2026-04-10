from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.load_data import load_text_dataset
from src.tokenizers import build_tokenizer


DATASET_NAMES = ("one-billion-word", "wikitext-103", "text8", "enwik8")
TOKENIZER_NAMES = ("char", "word", "bpe")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "outputs" / "tokenizer_compare"


def validate_dataset_name(dataset_name: str) -> None:
    if dataset_name not in DATASET_NAMES:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Expected one of: {', '.join(DATASET_NAMES)}"
        )


def limit_items(items, max_items: int | None):
    if max_items is None:
        yield from items
        return

    for index, item in enumerate(items):
        if index >= max_items:
            break
        yield item


def load_texts_from_dataset(dataset_name: str, split_name: str, max_texts: int | None) -> list[str]:
    validate_dataset_name(dataset_name)
    text_dataset = load_text_dataset(dataset_name)

    if split_name not in text_dataset.split_names:
        raise ValueError(
            f"Split '{split_name}' not found for dataset '{dataset_name}'. "
            f"Available splits: {', '.join(text_dataset.split_names)}"
        )

    return list(limit_items(text_dataset.iter_texts(split_name), max_texts))


def load_texts_from_file(file_path: Path, max_texts: int | None) -> list[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    texts = file_path.read_text(encoding="utf-8").splitlines()
    return list(limit_items((text for text in texts if text), max_texts))


def build_preview(tokens: list[str], limit: int = 20) -> list[str]:
    return tokens[:limit]


def evaluate_tokenizer(tokenizer_name: str, texts: list[str], max_vocab_size: int | None) -> dict:
    tokenizer = build_tokenizer(tokenizer_name, max_vocab_size=max_vocab_size)
    tokenizer.fit_from_texts(texts)

    encoded_ids = tokenizer.encode_texts(texts)
    decoded_tokens = tokenizer.decode_ids(encoded_ids[:20]) if encoded_ids else []

    if tokenizer_name == "bpe":
        preview_tokens = decoded_tokens
    else:
        preview_tokens = build_preview(list(tokenizer.iter_tokens_from_texts(texts)))

    text_lengths = [len(text) for text in texts]
    encoded_lengths = [len(tokenizer.encode_texts([text])) for text in texts if text]

    return {
        "tokenizer": tokenizer_name,
        "num_texts": len(texts),
        "vocab_size": tokenizer.vocab_size,
        "num_encoded_tokens": len(encoded_ids),
        "avg_chars_per_text": (sum(text_lengths) / len(text_lengths)) if text_lengths else 0.0,
        "avg_tokens_per_text": (sum(encoded_lengths) / len(encoded_lengths)) if encoded_lengths else 0.0,
        "preview_tokens": preview_tokens,
        "preview_token_ids": encoded_ids[:20],
    }


def compare_tokenizers(texts: list[str], tokenizer_names: list[str], max_vocab_size: int | None) -> list[dict]:
    results: list[dict] = []

    for tokenizer_name in tokenizer_names:
        try:
            results.append(evaluate_tokenizer(tokenizer_name, texts, max_vocab_size))
        except ImportError as exc:
            results.append(
                {
                    "tokenizer": tokenizer_name,
                    "error": str(exc),
                }
            )

    return results


def save_results(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare char, word, and BPE tokenizers on sample texts.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--dataset", type=str, default=None)
    source_group.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--tokenizers",
        type=str,
        nargs="+",
        default=list(TOKENIZER_NAMES),
        choices=TOKENIZER_NAMES,
        help="Choose one or more tokenizers to compare.",
    )
    parser.add_argument("--max-texts", type=int, default=10)
    parser.add_argument("--max-vocab-size", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset is not None:
        texts = load_texts_from_dataset(args.dataset, args.split, args.max_texts)
        output_name = f"{args.dataset}_{args.split}.json"
    else:
        texts = load_texts_from_file(args.text_file, args.max_texts)
        output_name = f"{args.text_file.stem}.json"

    if not texts:
        raise ValueError("No texts found for comparison.")

    results = compare_tokenizers(texts, args.tokenizers, args.max_vocab_size)
    output_path = args.output_dir / output_name
    save_results(results, output_path)

    print(f"Saved tokenizer comparison to: {output_path}")
    for result in results:
        if "error" in result:
            print(f"- {result['tokenizer']}: error -> {result['error']}")
            continue

        print(
            f"- {result['tokenizer']}: "
            f"vocab_size={result['vocab_size']}, "
            f"num_encoded_tokens={result['num_encoded_tokens']}, "
            f"avg_tokens_per_text={result['avg_tokens_per_text']:.2f}"
        )


if __name__ == "__main__":
    main()
