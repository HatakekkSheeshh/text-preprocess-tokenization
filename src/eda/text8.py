import csv
import json
import string
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from datasets import load_from_disk  # type: ignore
from matplotlib.ticker import FuncFormatter, PercentFormatter
from tqdm import tqdm


DATASET_NAME = "text8"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda" / DATASET_NAME
PLOTS_DIR = OUTPUT_DIR / "plots"
DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "figures" / DATASET_NAME

CHUNK_SIZE = 1_000_000
TOP_K_WORDS = 100
TOP_K_EXAMPLES = 20
TOP_K_WORDS_PLOT = 20
TOP_COVERAGE_RANKS = (10, 100, 1_000)
VOCAB_COVERAGE_PLOT_MAX_RANK = 10_000
RARE_WORD_MAX_FREQ = 2
WINDOW_SIZES = (128, 256, 512)
PUNCTUATION_CHARS = set(string.punctuation)
PLOT_FACE_COLOR = "#FFFDF7"
PLOT_ACCENT = "#C95F44"
PLOT_ACCENT_ALT = "#2B5F75"
PLOT_NEUTRAL = "#D6C6A8"
PLOT_TEXT = "#2A221C"
PLOT_GRID = "#E7DCCD"


def iter_chunk_words(text: str, chunk_size: int = CHUNK_SIZE):
    remainder = ""

    for start in tqdm(
        range(0, len(text), chunk_size),
        desc="Tokenizing chunks",
        unit="chunk",
        leave=False,
    ):
        chunk = remainder + text[start : start + chunk_size]

        if start + chunk_size < len(text):
            split_at = chunk.rfind(" ")
            if split_at == -1:
                remainder = chunk
                continue

            current = chunk[:split_at]
            remainder = chunk[split_at + 1 :]
        else:
            current = chunk
            remainder = ""

        for word in current.split():
            yield word

    if remainder:
        yield remainder


def percentile_from_counter(length_counter: Counter, percentile: float) -> int:
    if not length_counter:
        return 0

    total = sum(length_counter.values())
    threshold = total * percentile
    running_total = 0

    for length in sorted(length_counter):
        running_total += length_counter[length]
        if running_total >= threshold:
            return int(length)

    return int(max(length_counter))


def summarize_length_counter(length_counter: Counter) -> dict:
    if not length_counter:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0,
            "p95": 0,
        }

    total_items = sum(length_counter.values())
    weighted_total = sum(length * count for length, count in length_counter.items())

    return {
        "count": total_items,
        "min": int(min(length_counter)),
        "max": int(max(length_counter)),
        "mean": weighted_total / total_items,
        "median": percentile_from_counter(length_counter, 0.5),
        "p95": percentile_from_counter(length_counter, 0.95),
    }


def build_top_coverage(vocab_counter: Counter, total_words: int) -> dict:
    coverage = {}
    cumulative = 0
    max_rank = max(TOP_COVERAGE_RANKS)

    for rank, (_, frequency) in enumerate(vocab_counter.most_common(max_rank), start=1):
        cumulative += frequency
        if rank in TOP_COVERAGE_RANKS:
            coverage[f"top_{rank}"] = cumulative / total_words if total_words else 0.0

    return coverage


def analyze_split(text: str) -> tuple[dict, Counter, Counter, Counter]:
    char_counter = Counter(text)
    vocab_counter = Counter()
    word_length_counter = Counter()

    total_words = 0
    rare_word_count = 0
    total_word_characters = 0

    for word in iter_chunk_words(text):
        vocab_counter[word] += 1
        total_words += 1

        word_length = len(word)
        total_word_characters += word_length
        word_length_counter[word_length] += 1

    for frequency in vocab_counter.values():
        if frequency <= RARE_WORD_MAX_FREQ:
            rare_word_count += 1

    total_characters = len(text)
    space_count = char_counter.get(" ", 0)
    non_space_characters = total_characters - space_count
    punctuation_count = sum(count for ch, count in char_counter.items() if ch in PUNCTUATION_CHARS)
    digit_count = sum(count for ch, count in char_counter.items() if ch.isdigit())
    uppercase_count = sum(count for ch, count in char_counter.items() if ch.isupper())
    newline_count = char_counter.get("\n", 0)
    tab_count = char_counter.get("\t", 0)

    summary = {
        "num_rows": 1,
        "num_words": total_words,
        "vocab_size": len(vocab_counter),
        "character_vocab_size": len(char_counter),
        "total_characters": total_characters,
        "space_count": space_count,
        "non_space_characters": non_space_characters,
        "space_ratio": space_count / total_characters if total_characters else 0.0,
        "punctuation_count": punctuation_count,
        "punctuation_ratio": punctuation_count / total_characters if total_characters else 0.0,
        "digit_count": digit_count,
        "digit_ratio": digit_count / total_characters if total_characters else 0.0,
        "uppercase_count": uppercase_count,
        "uppercase_ratio": uppercase_count / total_characters if total_characters else 0.0,
        "newline_count": newline_count,
        "tab_count": tab_count,
        "double_space_count": text.count("  "),
        "avg_word_length": total_word_characters / total_words if total_words else 0.0,
        "lexical_diversity": len(vocab_counter) / total_words if total_words else 0.0,
        "num_rare_words": rare_word_count,
        "rare_word_ratio": rare_word_count / len(vocab_counter) if vocab_counter else 0.0,
        "word_length": summarize_length_counter(word_length_counter),
        "top_word_coverage": build_top_coverage(vocab_counter, total_words),
        "non_overlapping_windows": {
            str(window_size): total_words // window_size for window_size in WINDOW_SIZES
        },
        "character_set": sorted(repr(char)[1:-1] for char in char_counter.keys()),
    }

    return summary, vocab_counter, char_counter, word_length_counter


def build_oov_summary(train_vocab: set[str], eval_vocab_counter: Counter, total_words: int) -> dict:
    oov_counter = Counter(
        {word: freq for word, freq in eval_vocab_counter.items() if word not in train_vocab}
    )
    oov_token_count = sum(oov_counter.values())

    return {
        "unique_oov_words": len(oov_counter),
        "unique_oov_ratio": len(oov_counter) / len(eval_vocab_counter) if eval_vocab_counter else 0.0,
        "token_oov_count": oov_token_count,
        "token_oov_ratio": oov_token_count / total_words if total_words else 0.0,
        "in_vocab_token_ratio": 1.0 - (oov_token_count / total_words if total_words else 0.0),
    }


def save_json(summary: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_split_stats_csv(summary: dict) -> None:
    output_path = OUTPUT_DIR / "split_stats.csv"
    fieldnames = [
        "split",
        "num_words",
        "vocab_size",
        "character_vocab_size",
        "total_characters",
        "space_ratio",
        "punctuation_ratio",
        "digit_ratio",
        "uppercase_ratio",
        "avg_word_length",
        "lexical_diversity",
        "num_rare_words",
        "rare_word_ratio",
        "top_10_coverage",
        "top_100_coverage",
        "top_1000_coverage",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for split_name, split_summary in summary["splits"].items():
            writer.writerow(
                {
                    "split": split_name,
                    "num_words": split_summary["num_words"],
                    "vocab_size": split_summary["vocab_size"],
                    "character_vocab_size": split_summary["character_vocab_size"],
                    "total_characters": split_summary["total_characters"],
                    "space_ratio": split_summary["space_ratio"],
                    "punctuation_ratio": split_summary["punctuation_ratio"],
                    "digit_ratio": split_summary["digit_ratio"],
                    "uppercase_ratio": split_summary["uppercase_ratio"],
                    "avg_word_length": split_summary["avg_word_length"],
                    "lexical_diversity": split_summary["lexical_diversity"],
                    "num_rare_words": split_summary["num_rare_words"],
                    "rare_word_ratio": split_summary["rare_word_ratio"],
                    "top_10_coverage": split_summary["top_word_coverage"].get("top_10", 0.0),
                    "top_100_coverage": split_summary["top_word_coverage"].get("top_100", 0.0),
                    "top_1000_coverage": split_summary["top_word_coverage"].get("top_1000", 0.0),
                }
            )


def save_counter_csv(counter: Counter, file_name: str, header: tuple[str, str], limit: int | None = None) -> None:
    output_path = OUTPUT_DIR / file_name
    rows = counter.most_common(limit) if limit is not None else counter.most_common()

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(list(header))
        writer.writerows(rows)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_number(value: int | float) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.2f}"


def compact_number(value: float, _position: float) -> str:
    absolute_value = abs(value)
    if absolute_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute_value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def style_axis(
    ax,
    *,
    y_percent_axis: bool = False,
    y_numeric_axis: bool = True,
    x_percent_axis: bool = False,
    x_numeric_axis: bool = False,
    grid_axis: str = "y",
) -> None:
    ax.set_facecolor(PLOT_FACE_COLOR)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.8, color=PLOT_GRID, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PLOT_GRID)
    ax.spines["bottom"].set_color(PLOT_GRID)
    ax.tick_params(colors=PLOT_TEXT)
    ax.xaxis.label.set_color(PLOT_TEXT)
    ax.yaxis.label.set_color(PLOT_TEXT)
    ax.title.set_color(PLOT_TEXT)

    if y_percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    elif y_numeric_axis:
        ax.yaxis.set_major_formatter(FuncFormatter(compact_number))

    if x_percent_axis:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    elif x_numeric_axis:
        ax.xaxis.set_major_formatter(FuncFormatter(compact_number))


def make_figure_directories() -> list[Path]:
    directories = [PLOTS_DIR, DOCS_FIGURES_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def save_figure(fig: plt.Figure, file_name: str) -> None:
    for directory in make_figure_directories():
        fig.savefig(directory / file_name, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def counter_to_records(counter: Counter, limit: int) -> list[dict]:
    return [{"word": word, "frequency": frequency} for word, frequency in counter.most_common(limit)]


def get_longest_word_records(counter: Counter, limit: int) -> list[dict]:
    sorted_items = sorted(counter.items(), key=lambda item: (-len(item[0]), -item[1], item[0]))
    return [
        {"word": word, "length": len(word), "frequency": frequency}
        for word, frequency in sorted_items[:limit]
    ]


def plot_split_overview(summary: dict) -> None:
    splits = ["train", "validation", "test"]
    split_labels = [split.title() for split in splits]
    num_words = [summary["splits"][split]["num_words"] for split in splits]
    vocab_sizes = [summary["splits"][split]["vocab_size"] for split in splits]
    rare_ratios = [summary["splits"][split]["rare_word_ratio"] for split in splits]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=PLOT_FACE_COLOR)
    fig.suptitle("Text8 Split Overview", fontsize=16, fontweight="bold", color=PLOT_TEXT)

    axes[0].bar(split_labels, num_words, color=[PLOT_ACCENT, PLOT_ACCENT_ALT, PLOT_NEUTRAL])
    axes[0].set_title("Token Count by Split")
    axes[0].set_ylabel("Words")
    style_axis(axes[0])

    axes[1].bar(split_labels, vocab_sizes, color=[PLOT_ACCENT, PLOT_ACCENT_ALT, PLOT_NEUTRAL])
    axes[1].set_title("Vocabulary Size by Split")
    axes[1].set_ylabel("Unique Words")
    style_axis(axes[1])

    axes[2].bar(split_labels, rare_ratios, color=[PLOT_ACCENT, PLOT_ACCENT_ALT, PLOT_NEUTRAL])
    axes[2].set_title(f"Rare-Word Ratio (freq <= {RARE_WORD_MAX_FREQ})")
    axes[2].set_ylabel("Ratio")
    style_axis(axes[2], y_percent_axis=True)

    save_figure(fig, "split_overview.png")


def plot_top_words(train_vocab_counter: Counter) -> None:
    top_items = train_vocab_counter.most_common(TOP_K_WORDS_PLOT)
    words = [word for word, _ in top_items][::-1]
    frequencies = [frequency for _, frequency in top_items][::-1]
    positions = list(range(len(words)))

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=PLOT_FACE_COLOR)
    ax.barh(positions, frequencies, color=PLOT_ACCENT)
    ax.set_yticks(positions, labels=words)
    ax.set_title(f"Top {TOP_K_WORDS_PLOT} Most Frequent Words in Text8 Train", fontsize=15, fontweight="bold")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    style_axis(ax, y_numeric_axis=False, x_numeric_axis=True, grid_axis="x")

    save_figure(fig, "top_words.png")


def plot_word_length_distribution(word_length_counter: Counter, summary: dict) -> None:
    lengths = sorted(word_length_counter)
    counts = [word_length_counter[length] for length in lengths]
    train_length_summary = summary["splits"]["train"]["word_length"]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=PLOT_FACE_COLOR)
    ax.bar(lengths, counts, color=PLOT_ACCENT_ALT, width=0.85)
    ax.set_title("Word Length Distribution in Text8 Train", fontsize=15, fontweight="bold")
    ax.set_xlabel("Word Length")
    ax.set_ylabel("Word Count")
    ax.set_yscale("log")
    ax.axvline(train_length_summary["median"], color=PLOT_ACCENT, linestyle="--", linewidth=1.8, label="Median")
    ax.axvline(train_length_summary["p95"], color=PLOT_NEUTRAL, linestyle="--", linewidth=1.8, label="95th percentile")
    style_axis(ax)
    ax.legend(frameon=False)

    save_figure(fig, "word_length_distribution.png")


def plot_oov_summary(summary: dict) -> None:
    splits = ["validation", "test"]
    split_labels = [split.title() for split in splits]
    unique_oov_ratios = [summary["oov_vs_train"][split]["unique_oov_ratio"] for split in splits]
    token_oov_ratios = [summary["oov_vs_train"][split]["token_oov_ratio"] for split in splits]
    positions = list(range(len(splits)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.6, 5.3), facecolor=PLOT_FACE_COLOR)
    ax.bar([position - width / 2 for position in positions], unique_oov_ratios, width=width, color=PLOT_ACCENT, label="Unique OOV ratio")
    ax.bar([position + width / 2 for position in positions], token_oov_ratios, width=width, color=PLOT_ACCENT_ALT, label="Token OOV ratio")
    ax.set_xticks(positions, split_labels)
    ax.set_title("OOV Ratios Against Training Vocabulary", fontsize=15, fontweight="bold")
    ax.set_ylabel("Ratio")
    style_axis(ax, y_percent_axis=True)
    ax.legend(frameon=False)

    save_figure(fig, "oov_summary.png")


def plot_vocab_coverage_curve(train_vocab_counter: Counter, summary: dict) -> None:
    most_common_words = train_vocab_counter.most_common(VOCAB_COVERAGE_PLOT_MAX_RANK)
    total_words = summary["splits"]["train"]["num_words"]
    cumulative_frequency = 0
    ranks = []
    coverage_values = []

    for rank, (_, frequency) in enumerate(most_common_words, start=1):
        cumulative_frequency += frequency
        ranks.append(rank)
        coverage_values.append(cumulative_frequency / total_words if total_words else 0.0)

    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=PLOT_FACE_COLOR)
    ax.plot(ranks, coverage_values, color=PLOT_ACCENT, linewidth=2.5)
    ax.scatter(
        list(TOP_COVERAGE_RANKS),
        [summary["splits"]["train"]["top_word_coverage"][f"top_{rank}"] for rank in TOP_COVERAGE_RANKS],
        color=PLOT_ACCENT_ALT,
        s=40,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_title("Cumulative Token Coverage by Vocabulary Rank", fontsize=15, fontweight="bold")
    ax.set_xlabel("Top-k vocabulary rank (log scale)")
    ax.set_ylabel("Coverage")
    style_axis(ax, y_percent_axis=True)

    for rank in TOP_COVERAGE_RANKS:
        value = summary["splits"]["train"]["top_word_coverage"][f"top_{rank}"]
        ax.annotate(
            f"top {rank}: {value * 100:.1f}%",
            xy=(rank, value),
            xytext=(8, 8),
            textcoords="offset points",
            color=PLOT_TEXT,
            fontsize=9,
        )

    save_figure(fig, "vocab_coverage_curve.png")


def generate_plots(
    summary: dict,
    split_vocab_counters: dict[str, Counter],
    train_word_length_counter: Counter,
) -> None:
    plot_split_overview(summary)
    plot_top_words(split_vocab_counters["train"])
    plot_word_length_distribution(train_word_length_counter, summary)
    plot_oov_summary(summary)
    plot_vocab_coverage_curve(split_vocab_counters["train"], summary)


def build_report(summary: dict) -> str:
    train = summary["splits"]["train"]
    validation = summary["splits"]["validation"]
    test = summary["splits"]["test"]
    validation_oov = summary["oov_vs_train"]["validation"]
    test_oov = summary["oov_vs_train"]["test"]
    examples = summary["examples"]

    train_windows = train["non_overlapping_windows"]
    train_word_lengths = train["word_length"]
    top_train_words = ", ".join(item["word"] for item in examples["train_top_words"][:10])
    longest_examples = ", ".join(item["word"] for item in examples["train_longest_words"][:5])

    return f"""# Text8 EDA for Section 3.2

## 1. Dataset overview

Text8 is an aggressively normalized corpus derived from Wikipedia. In this repository version, each split is stored as a single continuous text stream:

- Train: {format_number(train["total_characters"])} characters, {format_number(train["num_words"])} whitespace-delimited words
- Validation: {format_number(validation["total_characters"])} characters, {format_number(validation["num_words"])} words
- Test: {format_number(test["total_characters"])} characters, {format_number(test["num_words"])} words

This already tells us that `text8` is structurally very different from datasets such as WikiText-103:

- there are no sentence boundaries to rely on
- there is no punctuation to exploit
- there is no uppercase information
- the corpus is almost a pure stream of lowercase words separated by spaces

For section 3.2, this means the EDA should focus on token stream properties rather than sentence-level discourse properties.

## 2. Surface-form characteristics

- Character vocabulary size in every split is only {train["character_vocab_size"]}, which is consistent with a nearly closed alphabet.
- Punctuation ratio is {format_percent(train["punctuation_ratio"])}, uppercase ratio is {format_percent(train["uppercase_ratio"])}, and digit ratio is {format_percent(train["digit_ratio"])} on the training split.
- Space ratio is {format_percent(train["space_ratio"])}, so the corpus alternates very regularly between alphabetic spans and separators.
- The character set is: {", ".join(train["character_set"])}.
- `double_space_count`, `newline_count`, and `tab_count` are all useful sanity checks for normalization quality. On the training split they are {train["double_space_count"]}, {train["newline_count"]}, and {train["tab_count"]}.

Interpretation:

`text8` is already preprocessed extremely heavily before it reaches our pipeline. Because of that, standard cleaning operations such as lowercasing, punctuation removal, or number normalization are either redundant or actively harmful for reproducibility. The right preprocessing strategy here is minimalism.

## 3. Word-level statistics

- Training vocabulary size: {format_number(train["vocab_size"])}
- Validation vocabulary size: {format_number(validation["vocab_size"])}
- Test vocabulary size: {format_number(test["vocab_size"])}
- Average word length on train: {format_number(train["avg_word_length"])} characters
- Median word length on train: {train_word_lengths["median"]}
- 95th percentile word length on train: {train_word_lengths["p95"]}
- Maximum observed word length on train: {train_word_lengths["max"]}
- Lexical diversity on train (`vocab_size / num_words`): {format_number(train["lexical_diversity"])}
- Rare-word ratio on train (frequency <= {RARE_WORD_MAX_FREQ}): {format_percent(train["rare_word_ratio"])}

The top-word coverage is also informative:

- Top 10 most frequent words cover {format_percent(train["top_word_coverage"]["top_10"])} of all training tokens
- Top 100 cover {format_percent(train["top_word_coverage"]["top_100"])}
- Top 1,000 cover {format_percent(train["top_word_coverage"]["top_1000"])}

Interpretation:

The corpus is still large enough to have a long-tail vocabulary, but it is much cleaner and more repetitive than raw Wikipedia text. This usually helps token-based language models converge faster because there are fewer orthographic variants competing for probability mass.

## 4. Notable normalization artifacts

- The most frequent training words start with: {top_train_words}.
- Number words such as `one`, `zero`, `nine`, `two`, `eight`, and `five` appear among the most frequent tokens, showing that numeric content survives mostly in verbalized form rather than digit form.
- The standalone token `s` is highly frequent, which strongly suggests that apostrophe-based forms were normalized in a way that detached possessive or contraction remnants.
- The longest observed tokens include: {longest_examples}.

Interpretation:

These examples show that `text8` is not raw natural text. It is a benchmark-oriented normalized stream. That is useful because it reduces surface noise, but it also means some artifacts are baked into the benchmark itself. For a word-level model, those unusually long merged strings inflate the vocabulary tail. For BPE, they are easier to decompose into reusable subword units.

## 5. Split overlap and OOV behavior

Compared with the training vocabulary:

- Validation unique OOV ratio: {format_percent(validation_oov["unique_oov_ratio"])}
- Validation token-level OOV ratio: {format_percent(validation_oov["token_oov_ratio"])}
- Test unique OOV ratio: {format_percent(test_oov["unique_oov_ratio"])}
- Test token-level OOV ratio: {format_percent(test_oov["token_oov_ratio"])}

Interpretation:

Even in a normalized benchmark like `text8`, word-level modeling does not fully escape OOV. The important nuance is whether OOV is concentrated in many rare types or in a meaningful portion of evaluation tokens. If token-level OOV stays low, word-level models can remain competitive despite unseen vocabulary. If it rises, subword tokenization becomes more attractive.

## 6. Implications for preprocessing

Recommended preprocessing for `text8`:

1. Preserve the existing lowercase stream exactly as provided.
2. Tokenize words using whitespace only for the word-level baseline.
3. Keep the space character as a valid token for character-level modeling.
4. Train BPE directly on the already normalized training split; do not add extra normalization unless the whole team agrees to deviate from the benchmark.
5. Build training samples from fixed contiguous windows instead of sentences, because sentence segmentation is not meaningful here.

For sequence construction, the training split alone can produce approximately:

- {format_number(train_windows["128"])} non-overlapping windows of 128 words
- {format_number(train_windows["256"])} non-overlapping windows of 256 words
- {format_number(train_windows["512"])} non-overlapping windows of 512 words

This is a useful design signal for your dataloader and for estimating training time.

## 7. Expected model behavior

### Word-level

Word-level tokenization is likely to be stronger on `text8` than on more raw datasets because the corpus has already removed case, punctuation, and many formatting artifacts. The main weakness is the still-large softmax vocabulary and the remaining OOV words on validation/test.

### Character-level

Character-level modeling will enjoy an extremely small vocabulary and essentially no OOV problem. However, it will need much longer sequences to represent the same amount of information. On `text8`, that usually means slower training and harder long-range dependency learning, especially for small RNN/LSTM models.

### BPE

BPE is likely to be the best compromise. Because `text8` is already normalized, the gain from BPE may be smaller than on noisier corpora, but it should still reduce vocabulary size substantially while avoiding much of the OOV burden of pure word-level modeling.

## 8. Practical hypothesis for the report

If the same model family and capacity are used across tokenizers, a reasonable hypothesis for `text8` is:

- BPE will provide the best trade-off between perplexity and efficiency.
- Word-level may achieve competitive perplexity if the vocabulary cap is generous, but it will be more expensive in the embedding/softmax layers.
- Character-level will be the most robust to unseen forms but probably the slowest and weakest in perplexity unless trained longer or with a stronger architecture.

This hypothesis follows directly from the dataset structure: `text8` is simple at the surface level, but still long-context and vocabulary-heavy enough that token granularity matters.
"""


def save_report(summary: dict) -> None:
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text(build_report(summary), encoding="utf-8")


def run_text8_eda() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Please run `python src/main.py --download text8` first."
        )

    dataset_dict = load_from_disk(str(DATASET_PATH))

    split_summaries = {}
    split_vocab_counters = {}
    split_char_counters = {}
    train_word_length_counter = Counter()

    for split_name in tqdm(dataset_dict.keys(), desc="Analyzing splits", unit="split"):
        text = dataset_dict[split_name][0]["text"]
        split_summary, vocab_counter, char_counter, word_length_counter = analyze_split(text)
        split_summaries[split_name] = split_summary
        split_vocab_counters[split_name] = vocab_counter
        split_char_counters[split_name] = char_counter

        if split_name == "train":
            train_word_length_counter = word_length_counter

    train_vocab = set(split_vocab_counters["train"])
    oov_vs_train = {
        split_name: build_oov_summary(
            train_vocab,
            split_vocab_counters[split_name],
            split_summaries[split_name]["num_words"],
        )
        for split_name in ("validation", "test")
    }

    summary = {
        "dataset": DATASET_NAME,
        "notes": {
            "structure": "Each split is a single continuous normalized text stream.",
            "recommended_preprocessing": "Avoid extra normalization; whitespace tokenization is enough for the word-level baseline.",
        },
        "splits": split_summaries,
        "oov_vs_train": oov_vs_train,
        "examples": {
            "train_top_words": counter_to_records(split_vocab_counters["train"], TOP_K_EXAMPLES),
            "train_longest_words": get_longest_word_records(split_vocab_counters["train"], TOP_K_EXAMPLES),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(summary)
    save_split_stats_csv(summary)
    save_counter_csv(split_vocab_counters["train"], "train_vocab_top100.csv", ("word", "frequency"), limit=TOP_K_WORDS)
    save_counter_csv(split_char_counters["train"], "train_character_distribution.csv", ("character", "frequency"))
    save_counter_csv(train_word_length_counter, "train_word_length_distribution.csv", ("word_length", "count"))
    longest_words_path = OUTPUT_DIR / "train_longest_words.csv"
    with longest_words_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["word", "length", "frequency"])
        writer.writeheader()
        writer.writerows(summary["examples"]["train_longest_words"])
    save_counter_csv(
        Counter({word: freq for word, freq in split_vocab_counters["validation"].items() if word not in train_vocab}),
        "validation_oov_words.csv",
        ("word", "frequency"),
    )
    save_counter_csv(
        Counter({word: freq for word, freq in split_vocab_counters["test"].items() if word not in train_vocab}),
        "test_oov_words.csv",
        ("word", "frequency"),
    )
    save_report(summary)
    generate_plots(summary, split_vocab_counters, train_word_length_counter)

    return summary


if __name__ == "__main__":
    eda_summary = run_text8_eda()
    print(json.dumps(eda_summary, indent=2))
