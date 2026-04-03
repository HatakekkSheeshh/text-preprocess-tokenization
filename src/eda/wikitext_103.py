import csv
import json
import re
import string
from collections import Counter
from pathlib import Path

from datasets import load_from_disk  # type: ignore


DATASET_NAME = "wikitext-103"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda" / DATASET_NAME

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
HEADING_PATTERN = re.compile(r"^\s*=+\s*[^=].*?[^=]\s*=+\s*$")
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
PUNCTUATION_CHARS = set(string.punctuation)

TOP_K_WORDS = 100
RARE_WORD_MAX_FREQ = 2


def tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def is_heading(text: str) -> bool:
    return bool(HEADING_PATTERN.match(text.strip()))


def split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(text) if segment.strip()]


def init_split_stats() -> dict:
    return {
        "num_rows": 0,
        "num_non_empty_rows": 0,
        "num_documents": 0,
        "num_sentences": 0,
        "num_words": 0,
        "num_characters": 0,
        "num_non_space_characters": 0,
        "num_punctuation_characters": 0,
        "document_lengths": [],
        "sentence_lengths": [],
        "row_lengths": [],
    }


def analyze_split(split_ds) -> tuple[dict, Counter]:
    stats = init_split_stats()
    vocab_counter = Counter()

    current_document_length = 0

    for row in split_ds:
        text = row["text"]
        stats["num_rows"] += 1

        if not text or not text.strip():
            if current_document_length > 0:
                stats["document_lengths"].append(current_document_length)
                stats["num_documents"] += 1
                current_document_length = 0
            continue

        if is_heading(text):
            if current_document_length > 0:
                stats["document_lengths"].append(current_document_length)
                stats["num_documents"] += 1
                current_document_length = 0
            continue

        stats["num_non_empty_rows"] += 1
        stats["num_characters"] += len(text)

        non_space_chars = [ch for ch in text if not ch.isspace()]
        stats["num_non_space_characters"] += len(non_space_chars)
        stats["num_punctuation_characters"] += sum(ch in PUNCTUATION_CHARS for ch in non_space_chars)

        words = tokenize_words(text)
        word_count = len(words)
        stats["num_words"] += word_count
        stats["row_lengths"].append(word_count)
        current_document_length += word_count
        vocab_counter.update(words)

        sentences = split_sentences(text)
        for sentence in sentences:
            sentence_len = len(tokenize_words(sentence))
            if sentence_len > 0:
                stats["sentence_lengths"].append(sentence_len)
                stats["num_sentences"] += 1

    if current_document_length > 0:
        stats["document_lengths"].append(current_document_length)
        stats["num_documents"] += 1

    return stats, vocab_counter


def summarize_lengths(values: list[int]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
        }

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def build_split_summary(stats: dict, vocab_counter: Counter) -> dict:
    punctuation_ratio = 0.0
    if stats["num_non_space_characters"] > 0:
        punctuation_ratio = stats["num_punctuation_characters"] / stats["num_non_space_characters"]

    rare_words = {word: freq for word, freq in vocab_counter.items() if freq <= RARE_WORD_MAX_FREQ}

    return {
        "num_rows": stats["num_rows"],
        "num_non_empty_rows": stats["num_non_empty_rows"],
        "num_documents": stats["num_documents"],
        "num_sentences": stats["num_sentences"],
        "num_words": stats["num_words"],
        "vocab_size": len(vocab_counter),
        "punctuation_ratio": punctuation_ratio,
        "rare_word_threshold": RARE_WORD_MAX_FREQ,
        "num_rare_words": len(rare_words),
        "rare_word_ratio": len(rare_words) / len(vocab_counter) if vocab_counter else 0.0,
        "document_length": summarize_lengths(stats["document_lengths"]),
        "sentence_length": summarize_lengths(stats["sentence_lengths"]),
        "row_length": summarize_lengths(stats["row_lengths"]),
    }


def save_summary(summary: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_split_stats_csv(summary: dict) -> None:
    output_path = OUTPUT_DIR / "split_stats.csv"
    fieldnames = [
        "split",
        "num_rows",
        "num_non_empty_rows",
        "num_documents",
        "num_sentences",
        "num_words",
        "vocab_size",
        "punctuation_ratio",
        "num_rare_words",
        "rare_word_ratio",
        "avg_document_length",
        "avg_sentence_length",
        "avg_row_length",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, split_summary in summary.items():
            writer.writerow(
                {
                    "split": split_name,
                    "num_rows": split_summary["num_rows"],
                    "num_non_empty_rows": split_summary["num_non_empty_rows"],
                    "num_documents": split_summary["num_documents"],
                    "num_sentences": split_summary["num_sentences"],
                    "num_words": split_summary["num_words"],
                    "vocab_size": split_summary["vocab_size"],
                    "punctuation_ratio": split_summary["punctuation_ratio"],
                    "num_rare_words": split_summary["num_rare_words"],
                    "rare_word_ratio": split_summary["rare_word_ratio"],
                    "avg_document_length": split_summary["document_length"]["mean"],
                    "avg_sentence_length": split_summary["sentence_length"]["mean"],
                    "avg_row_length": split_summary["row_length"]["mean"],
                }
            )


def save_vocab_csv(vocab_counter: Counter, file_name: str, limit: int | None = None) -> None:
    output_path = OUTPUT_DIR / file_name
    items = vocab_counter.most_common(limit) if limit is not None else vocab_counter.most_common()

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["word", "frequency"])
        writer.writerows(items)


def run_wikitext_103_eda() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Please download and save it to data/raw/wikitext-103 first."
        )

    dataset_dict = load_from_disk(str(DATASET_PATH))

    summary = {}
    train_vocab_counter = Counter()

    for split_name in dataset_dict.keys():
        split_stats, vocab_counter = analyze_split(dataset_dict[split_name])
        summary[split_name] = build_split_summary(split_stats, vocab_counter)
        if split_name == "train":
            train_vocab_counter = vocab_counter

    save_summary(summary)
    save_split_stats_csv(summary)
    save_vocab_csv(train_vocab_counter, "train_vocab_top100.csv", limit=TOP_K_WORDS)

    rare_vocab_counter = Counter(
        {
            word: freq
            for word, freq in train_vocab_counter.items()
            if freq <= RARE_WORD_MAX_FREQ
        }
    )
    save_vocab_csv(rare_vocab_counter, "train_rare_words.csv")

    return summary


if __name__ == "__main__":
    eda_summary = run_wikitext_103_eda()
    print(json.dumps(eda_summary, indent=2))
