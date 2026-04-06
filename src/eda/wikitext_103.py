import csv
import json
import re
import string
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_from_disk  # type: ignore
from tqdm import tqdm


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


# -----------------------------
# Text processing helpers
# -----------------------------

def tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def is_heading(text: str) -> bool:
    return bool(HEADING_PATTERN.match(text.strip()))


def split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(text) if segment.strip()]


# -----------------------------
# EDA/statistics helpers
# -----------------------------

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


def close_current_document(stats: dict, current_document_length: int, has_current_document: bool) -> None:
    if has_current_document:
        stats["document_lengths"].append(current_document_length)


def update_text_statistics(stats: dict, text: str, vocab_counter: Counter, current_document_length: int) -> int:
    stats["num_non_empty_rows"] += 1
    stats["num_characters"] += len(text)

    non_space_chars = [ch for ch in text if not ch.isspace()]
    stats["num_non_space_characters"] += len(non_space_chars)
    stats["num_punctuation_characters"] += sum(
        ch in PUNCTUATION_CHARS for ch in non_space_chars
    )

    words = tokenize_words(text)
    word_count = len(words)
    stats["num_words"] += word_count
    stats["row_lengths"].append(word_count)
    vocab_counter.update(words)

    for sentence in split_sentences(text):
        sentence_len = len(tokenize_words(sentence))
        if sentence_len > 0:
            stats["sentence_lengths"].append(sentence_len)
            stats["num_sentences"] += 1

    return current_document_length + word_count


def analyze_split(split_name: str, split_ds) -> tuple[dict, Counter]:
    stats = init_split_stats()
    vocab_counter = Counter()

    current_document_length = 0
    has_current_document = False

    for row in tqdm(split_ds, desc=f"Analyzing {split_name}", unit="rows"):
        text = row["text"]
        stats["num_rows"] += 1

        if not text or not text.strip():
            continue

        if is_heading(text):
            close_current_document(stats, current_document_length, has_current_document)
            stats["num_documents"] += 1
            current_document_length = 0
            has_current_document = True
            continue

        current_document_length = update_text_statistics(
            stats=stats,
            text=text,
            vocab_counter=vocab_counter,
            current_document_length=current_document_length,
        )

    close_current_document(stats, current_document_length, has_current_document)

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


# -----------------------------
# Output saving helpers
# -----------------------------

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_summary(summary: dict) -> None:
    ensure_output_dir()
    output_path = OUTPUT_DIR / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_split_stats_csv(summary: dict) -> None:
    ensure_output_dir()
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
    ensure_output_dir()
    output_path = OUTPUT_DIR / file_name
    items = vocab_counter.most_common(limit) if limit is not None else vocab_counter.most_common()

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["word", "frequency"])
        writer.writerows(items)


def save_length_histogram(values: list[int], title: str, xlabel: str, file_name: str, max_x: int | None = None) -> None:
    ensure_output_dir()
    plt.figure(figsize=(10, 6))

    if max_x is not None:
        values = [value for value in values if value <= max_x]

    plt.hist(values, bins=50, color="#4C72B0", edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / file_name, dpi=200)
    plt.close()


def save_split_bar_chart(summary: dict, metric_name: str, title: str, ylabel: str, file_name: str) -> None:
    ensure_output_dir()
    split_names = list(summary.keys())
    values = [summary[split_name][metric_name] for split_name in split_names]

    plt.figure(figsize=(8, 5))
    plt.bar(split_names, values, color="#55A868", edgecolor="black")
    plt.title(title)
    plt.xlabel("Split")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / file_name, dpi=200)
    plt.close()


def save_top_words_plot(vocab_counter: Counter, top_k: int = 30) -> None:
    ensure_output_dir()
    top_words = vocab_counter.most_common(top_k)
    words = [word for word, _ in top_words]
    freqs = [freq for _, freq in top_words]

    plt.figure(figsize=(12, 6))
    plt.bar(words, freqs, color="#C44E52", edgecolor="black")
    plt.title(f"Top {top_k} Most Frequent Words in Train Split")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "train_top30_words.png", dpi=200)
    plt.close()


def save_zipf_plot(vocab_counter: Counter) -> None:
    ensure_output_dir()
    sorted_freqs = [freq for _, freq in vocab_counter.most_common()]
    ranks = list(range(1, len(sorted_freqs) + 1))

    plt.figure(figsize=(8, 6))
    plt.loglog(ranks, sorted_freqs, color="#8172B3")
    plt.title("Train Vocabulary Zipf Plot")
    plt.xlabel("Word Rank (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "train_vocab_zipf.png", dpi=200)
    plt.close()


def save_eda_plots(summary: dict, split_stats_map: dict, train_vocab_counter: Counter) -> None:
    save_split_bar_chart(
        summary=summary,
        metric_name="vocab_size",
        title="Vocabulary Size by Split",
        ylabel="Vocabulary Size",
        file_name="split_vocab_size.png",
    )
    save_split_bar_chart(
        summary=summary,
        metric_name="rare_word_ratio",
        title="Rare Word Ratio by Split",
        ylabel="Rare Word Ratio",
        file_name="split_rare_word_ratio.png",
    )
    save_split_bar_chart(
        summary=summary,
        metric_name="punctuation_ratio",
        title="Punctuation Ratio by Split",
        ylabel="Punctuation Ratio",
        file_name="split_punctuation_ratio.png",
    )

    train_stats = split_stats_map["train"]
    save_length_histogram(
        values=train_stats["document_lengths"],
        title="Train Document Length Distribution",
        xlabel="Document Length (words)",
        file_name="train_document_length_hist.png",
        max_x=2000,
    )
    save_length_histogram(
        values=train_stats["sentence_lengths"],
        title="Train Sentence Length Distribution",
        xlabel="Sentence Length (words)",
        file_name="train_sentence_length_hist.png",
        max_x=100,
    )
    save_top_words_plot(train_vocab_counter, top_k=30)
    save_zipf_plot(train_vocab_counter)


def save_eda_outputs(summary: dict, split_stats_map: dict, train_vocab_counter: Counter) -> None:
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
    save_eda_plots(summary, split_stats_map, train_vocab_counter)


# -----------------------------
# Main EDA runner
# -----------------------------

def run_wikitext_103_eda() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Please download and save it to data/raw/wikitext-103 first."
        )

    dataset_dict = load_from_disk(str(DATASET_PATH))

    summary = {}
    split_stats_map = {}
    train_vocab_counter = Counter()

    for split_name in dataset_dict.keys():
        split_stats, vocab_counter = analyze_split(split_name, dataset_dict[split_name])
        split_stats_map[split_name] = split_stats
        summary[split_name] = build_split_summary(split_stats, vocab_counter)
        if split_name == "train":
            train_vocab_counter = vocab_counter

    save_eda_outputs(summary, split_stats_map, train_vocab_counter)

    return summary


# if __name__ == "__main__":
    # eda_summary = run_wikitext_103_eda()
    # print(json.dumps(eda_summary, indent=2))
