import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from datasets import load_from_disk  # type: ignore
from tqdm import tqdm


DATASET_NAME = "one-billion-word"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda" / DATASET_NAME
PLOTS_DIR = OUTPUT_DIR / "plots"

TOKEN_RE = re.compile(r"\S+")
PUNCT_RE = re.compile(r"[^\w\s]")
DIGIT_RE = re.compile(r"\d")

SAMPLE_SIZE = 1_000_000
SMALL_SAMPLE_SIZE = 5_000
SEED = 42
TOP_K_TOKENS = 50
TOP_K_PLOT_TOKENS = 30
VOCAB_CAPS = (5_000, 10_000, 20_000, 30_000, 50_000, 100_000)
PLOT_FACE_COLOR = "#FFFDF7"
PLOT_ACCENT = "#C95F44"
PLOT_ACCENT_ALT = "#2B5F75"
PLOT_NEUTRAL = "#D6C6A8"
PLOT_TEXT = "#2A221C"
PLOT_GRID = "#E7DCCD"


def make_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(summary: dict) -> None:
    (OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_csv(file_name: str, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    with (OUTPUT_DIR / file_name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, file_name: str) -> None:
    fig.savefig(PLOTS_DIR / file_name, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def char_type(character: str) -> str:
    if character.isupper():
        return "upper"
    if character.islower():
        return "lower"
    if character.isdigit():
        return "digit"
    if character.isspace():
        return "space"
    if PUNCT_RE.match(character):
        return "punct"
    return "other"


def percentile(values: list[int], ratio: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * ratio)
    return float(sorted_values[index])


def estimate_oov_rows(sorted_tokens: list[tuple[str, int]], vocab_cap: int) -> dict:
    total_token_count = sum(freq for _, freq in sorted_tokens)
    top_vocab = {token for token, _ in sorted_tokens[:vocab_cap]}
    in_vocab = sum(freq for token, freq in sorted_tokens if token in top_vocab)
    oov_rate = 1.0 - (in_vocab / total_token_count if total_token_count else 0.0)

    return {
        "vocab_cap": vocab_cap,
        "estimated_oov_rate": oov_rate,
    }


def analyze_sample(sample) -> tuple[dict, Counter, Counter, Counter, list[int]]:
    token_counter = Counter()
    char_counter = Counter()
    char_type_counter = Counter()
    sentence_lengths = []

    num_rows = 0
    num_empty = 0
    num_with_digit = 0
    num_with_punctuation = 0

    for row in tqdm(sample, desc="Analyzing sample", unit="rows"):
        text = row["text"]
        num_rows += 1

        if not text or not text.strip():
            num_empty += 1
            continue

        normalized_text = re.sub(r"\s+", " ", text).strip()
        tokens = TOKEN_RE.findall(normalized_text)

        sentence_lengths.append(len(tokens))
        token_counter.update(tokens)
        char_counter.update(normalized_text)

        if DIGIT_RE.search(normalized_text):
            num_with_digit += 1
        if PUNCT_RE.search(normalized_text):
            num_with_punctuation += 1

    for character, count in char_counter.items():
        char_type_counter[char_type(character)] += count

    total_tokens = int(sum(token_counter.values()))
    vocab_size = len(token_counter)
    char_vocab_size = len(char_counter)

    summary = {
        "sample_rows": num_rows,
        "non_empty_rows": num_rows - num_empty,
        "empty_rows": num_empty,
        "total_tokens": total_tokens,
        "vocab_size_raw": vocab_size,
        "char_vocab_size": char_vocab_size,
        "avg_sentence_len_tokens": sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0,
        "median_sentence_len_tokens": percentile(sentence_lengths, 0.5),
        "p95_sentence_len_tokens": percentile(sentence_lengths, 0.95),
        "max_sentence_len_tokens": max(sentence_lengths) if sentence_lengths else 0,
        "pct_rows_with_digits": (num_with_digit / max(1, num_rows)) * 100,
        "pct_rows_with_punctuation": (num_with_punctuation / max(1, num_rows)) * 100,
    }

    return summary, token_counter, char_counter, char_type_counter, sentence_lengths


def save_summary_tables(summary: dict, token_counter: Counter, char_type_counter: Counter, sentence_lengths: list[int]) -> None:
    save_csv("summary_stats.csv", tuple(summary.keys()), [summary])

    freq_values = list(token_counter.values())
    rare_eq_1 = sum(1 for value in freq_values if value == 1)
    rare_lt_5 = sum(1 for value in freq_values if value < 5)
    vocab_size = len(token_counter)
    save_csv(
        "rare_token_stats.csv",
        ("vocab_size", "rare_freq_eq_1", "rare_freq_lt_5", "rare_eq_1_ratio", "rare_lt_5_ratio"),
        [
            {
                "vocab_size": vocab_size,
                "rare_freq_eq_1": rare_eq_1,
                "rare_freq_lt_5": rare_lt_5,
                "rare_eq_1_ratio": rare_eq_1 / max(1, vocab_size),
                "rare_lt_5_ratio": rare_lt_5 / max(1, vocab_size),
            }
        ],
    )

    save_csv(
        "top_tokens.csv",
        ("token", "frequency"),
        [{"token": token, "frequency": frequency} for token, frequency in token_counter.most_common(TOP_K_TOKENS)],
    )

    save_csv(
        "char_type_distribution.csv",
        ("char_type", "count", "ratio"),
        [
            {
                "char_type": char_group,
                "count": count,
                "ratio": count / max(1, sum(char_type_counter.values())),
            }
            for char_group, count in char_type_counter.most_common()
        ],
    )

    sorted_tokens = token_counter.most_common()
    save_csv(
        "oov_simulation_wordlevel.csv",
        ("vocab_cap", "estimated_oov_rate"),
        [estimate_oov_rows(sorted_tokens, vocab_cap) for vocab_cap in VOCAB_CAPS],
    )

    if sentence_lengths:
        save_csv(
            "token_length_compare_word_char_bpe.csv",
            ("tokenization", "avg_sequence_length"),
            [
                {"tokenization": "word-level", "avg_sequence_length": sum(sentence_lengths) / len(sentence_lengths)},
                {"tokenization": "char-level(no-space)", "avg_sequence_length": 0},
                {"tokenization": "bpe-8k(hypothesis)", "avg_sequence_length": 0},
                {"tokenization": "bpe-16k(hypothesis)", "avg_sequence_length": 0},
                {"tokenization": "bpe-32k(hypothesis)", "avg_sequence_length": 0},
            ],
        )


def save_sequence_comparison(sample) -> None:
    small_sample = sample.select(range(min(SMALL_SAMPLE_SIZE, len(sample))))
    word_lengths = []
    char_lengths = []

    for row in tqdm(small_sample, desc="Comparing sequence lengths", unit="rows", leave=False):
        text = row["text"]
        if not text or not text.strip():
            continue

        word_lengths.append(len(TOKEN_RE.findall(text)))
        char_lengths.append(len(text.replace(" ", "")))

    save_csv(
        "token_length_compare_word_char_bpe.csv",
        ("tokenization", "avg_sequence_length"),
        [
            {
                "tokenization": "word-level",
                "avg_sequence_length": sum(word_lengths) / len(word_lengths) if word_lengths else 0.0,
            },
            {
                "tokenization": "char-level(no-space)",
                "avg_sequence_length": sum(char_lengths) / len(char_lengths) if char_lengths else 0.0,
            },
            {"tokenization": "bpe-8k(hypothesis)", "avg_sequence_length": ""},
            {"tokenization": "bpe-16k(hypothesis)", "avg_sequence_length": ""},
            {"tokenization": "bpe-32k(hypothesis)", "avg_sequence_length": ""},
        ],
    )


def plot_document_char_distribution(sample) -> None:
    char_lengths = [
        len(row["text"])
        for row in tqdm(sample, desc="Collecting document lengths", unit="rows", leave=False)
        if row["text"]
    ]
    if not char_lengths:
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PLOT_FACE_COLOR)
    ax.hist(char_lengths, bins=100, color=PLOT_ACCENT, edgecolor=PLOT_FACE_COLOR)
    ax.set_xlim(0, 600)
    ax.set_xticks(list(range(0, 601, 50)))
    ax.set_title("One Billion Word Document Length Distribution", color=PLOT_TEXT)
    ax.set_xlabel("Characters per sentence", color=PLOT_TEXT)
    ax.set_ylabel("Count", color=PLOT_TEXT)
    ax.set_facecolor(PLOT_FACE_COLOR)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color=PLOT_GRID, alpha=0.85)
    ax.tick_params(colors=PLOT_TEXT)

    save_figure(fig, "document_char_distribution.png")


def plot_sentence_length_distribution(sentence_lengths: list[int]) -> None:
    if not sentence_lengths:
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PLOT_FACE_COLOR)
    ax.hist(sentence_lengths, bins=100, color=PLOT_ACCENT_ALT, edgecolor=PLOT_FACE_COLOR)
    ax.set_xlim(0, 200)
    ax.set_xticks(list(range(0, 201, 20)))
    ax.set_title("One Billion Word Sentence Length Distribution", color=PLOT_TEXT)
    ax.set_xlabel("Tokens per sentence", color=PLOT_TEXT)
    ax.set_ylabel("Count", color=PLOT_TEXT)
    ax.set_facecolor(PLOT_FACE_COLOR)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color=PLOT_GRID, alpha=0.85)
    ax.tick_params(colors=PLOT_TEXT)

    save_figure(fig, "length_distribution_200.png")


def plot_top_tokens(token_counter: Counter) -> None:
    top_tokens = token_counter.most_common(TOP_K_PLOT_TOKENS)
    if not top_tokens:
        return

    labels = [token for token, _ in reversed(top_tokens)]
    frequencies = [frequency for _, frequency in reversed(top_tokens)]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PLOT_FACE_COLOR)
    ax.barh(labels, frequencies, color=PLOT_ACCENT)
    ax.set_title("Top 30 Most Frequent Tokens", color=PLOT_TEXT)
    ax.set_xlabel("Frequency", color=PLOT_TEXT)
    ax.set_ylabel("Token", color=PLOT_TEXT)
    ax.set_facecolor(PLOT_FACE_COLOR)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, color=PLOT_GRID, alpha=0.85)
    ax.tick_params(colors=PLOT_TEXT)

    save_figure(fig, "top_tokens.png")


def plot_zipf(token_counter: Counter) -> None:
    frequencies = [frequency for _, frequency in token_counter.most_common()]
    if not frequencies:
        return

    ranks = list(range(1, len(frequencies) + 1))

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=PLOT_FACE_COLOR)
    ax.loglog(ranks, frequencies, color=PLOT_ACCENT_ALT)
    ax.set_title("Zipf Plot (log-rank vs log-frequency)", color=PLOT_TEXT)
    ax.set_xlabel("Rank (log scale)", color=PLOT_TEXT)
    ax.set_ylabel("Frequency (log scale)", color=PLOT_TEXT)
    ax.set_facecolor(PLOT_FACE_COLOR)
    ax.grid(which="both", linestyle="--", linewidth=0.8, color=PLOT_GRID, alpha=0.75)
    ax.tick_params(colors=PLOT_TEXT)

    save_figure(fig, "zipf_plot.png")


def plot_char_type_distribution(char_type_counter: Counter) -> None:
    if not char_type_counter:
        return

    labels = [label for label, _ in char_type_counter.most_common()]
    total = sum(char_type_counter.values())
    ratios = [count / total for _, count in char_type_counter.most_common()]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=PLOT_FACE_COLOR)
    ax.bar(labels, ratios, color=[PLOT_ACCENT, PLOT_ACCENT_ALT, PLOT_NEUTRAL, PLOT_ACCENT, PLOT_ACCENT_ALT][: len(labels)])
    ax.set_title("Character Type Ratio", color=PLOT_TEXT)
    ax.set_xlabel("Character Type", color=PLOT_TEXT)
    ax.set_ylabel("Ratio", color=PLOT_TEXT)
    ax.set_facecolor(PLOT_FACE_COLOR)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color=PLOT_GRID, alpha=0.85)
    ax.tick_params(colors=PLOT_TEXT)

    save_figure(fig, "char_type_distribution.png")


def run_one_billion_word_eda() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Please run `python main.py --load one-billion-word` first."
        )

    dataset_dict = load_from_disk(str(DATASET_PATH))
    train_split = dataset_dict["train"]
    sample = train_split.shuffle(seed=SEED).select(range(min(SAMPLE_SIZE, len(train_split))))

    make_directories()

    summary, token_counter, char_counter, char_type_counter, sentence_lengths = analyze_sample(sample)
    sorted_tokens = token_counter.most_common()

    summary["dataset"] = DATASET_NAME
    summary["sample_size_requested"] = SAMPLE_SIZE
    summary["sample_size_used"] = len(sample)
    summary["train_rows_total"] = len(train_split)
    summary["validation_rows_total"] = len(dataset_dict["validation"])
    summary["test_rows_total"] = len(dataset_dict["test"])
    summary["rare_token_stats"] = {
        "rare_freq_eq_1": sum(1 for value in token_counter.values() if value == 1),
        "rare_freq_lt_5": sum(1 for value in token_counter.values() if value < 5),
    }
    summary["top_tokens"] = [
        {"token": token, "frequency": frequency}
        for token, frequency in sorted_tokens[:TOP_K_TOKENS]
    ]
    summary["char_type_distribution"] = {
        label: count / max(1, sum(char_type_counter.values()))
        for label, count in char_type_counter.most_common()
    }
    summary["oov_simulation_wordlevel"] = [
        estimate_oov_rows(sorted_tokens, vocab_cap) for vocab_cap in VOCAB_CAPS
    ]
    summary["notes"] = {
        "sampling": "EDA is computed on a shuffled sample from the training split to keep runtime manageable.",
        "tokenization": "Whitespace tokenization is used to stay aligned with the benchmark's tokenized sentence files.",
    }

    save_json(summary)
    save_summary_tables(summary, token_counter, char_type_counter, sentence_lengths)
    save_sequence_comparison(sample)
    plot_document_char_distribution(sample)
    plot_sentence_length_distribution(sentence_lengths)
    plot_top_tokens(token_counter)
    plot_zipf(token_counter)
    plot_char_type_distribution(char_type_counter)

    return summary


if __name__ == "__main__":
    eda_summary = run_one_billion_word_eda()
    print(json.dumps(eda_summary, indent=2))
