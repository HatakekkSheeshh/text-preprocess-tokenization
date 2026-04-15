from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from collections import Counter
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from src.datasets.load_data import load_text_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics" / "train_word_counts"
DEFAULT_DATASETS = ("text8", "enwik8", "one-billion-word", "wikitext-103")
WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", flags=re.UNICODE)


@dataclass
class TrainWordCountMetrics:
    dataset_name: str
    split: str
    num_texts: int
    num_non_empty_texts: int
    num_characters: int
    num_words: int
    whitespace_tokens: int
    vocab_size: int
    avg_words_per_text: float
    lexical_diversity: float
    top_words: list[dict[str, int | str]]
    truncated: bool
    max_texts: int | None
    max_characters: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def iter_limited_texts(text_dataset, split_name: str, max_texts: int | None, max_characters: int | None):
    consumed_characters = 0

    for index, text in enumerate(text_dataset.iter_texts(split_name)):
        if max_texts is not None and index >= max_texts:
            return

        if max_characters is not None:
            remaining_characters = max_characters - consumed_characters
            if remaining_characters <= 0:
                return

            if len(text) > remaining_characters:
                yield text[:remaining_characters]
                return

        consumed_characters += len(text)
        yield text


def tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def count_train_words(
    dataset_name: str,
    *,
    max_texts: int | None = None,
    max_characters: int | None = None,
    top_k: int = 20,
) -> TrainWordCountMetrics:
    text_dataset = load_text_dataset(dataset_name)
    if "train" not in text_dataset.split_names:
        raise ValueError(
            f"Dataset '{dataset_name}' has no train split. Available splits: {', '.join(text_dataset.split_names)}"
        )

    vocab_counter: Counter[str] = Counter()
    num_texts = 0
    num_non_empty_texts = 0
    num_characters = 0
    whitespace_tokens = 0

    for text in tqdm(
        iter_limited_texts(text_dataset, "train", max_texts, max_characters),
        desc=f"Counting train words: {dataset_name}",
        unit="texts",
    ):
        num_texts += 1
        num_characters += len(text)

        stripped_text = text.strip()
        if stripped_text:
            num_non_empty_texts += 1
            whitespace_tokens += len(stripped_text.split())

        vocab_counter.update(tokenize_words(text))

    num_words = sum(vocab_counter.values())
    vocab_size = len(vocab_counter)
    avg_words_per_text = num_words / num_texts if num_texts else 0.0
    lexical_diversity = vocab_size / num_words if num_words else 0.0
    truncated = max_texts is not None or max_characters is not None

    return TrainWordCountMetrics(
        dataset_name=dataset_name,
        split="train",
        num_texts=num_texts,
        num_non_empty_texts=num_non_empty_texts,
        num_characters=num_characters,
        num_words=num_words,
        whitespace_tokens=whitespace_tokens,
        vocab_size=vocab_size,
        avg_words_per_text=avg_words_per_text,
        lexical_diversity=lexical_diversity,
        top_words=[
            {"word": word, "frequency": frequency}
            for word, frequency in vocab_counter.most_common(top_k)
        ],
        truncated=truncated,
        max_texts=max_texts,
        max_characters=max_characters,
    )


def compact_number(value: float, _position: float) -> str:
    absolute_value = abs(value)
    if absolute_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if absolute_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute_value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def save_csv(results: list[TrainWordCountMetrics], output_dir: Path) -> Path:
    output_path = output_dir / "train_word_counts.csv"
    fieldnames = [
        "dataset_name",
        "split",
        "num_texts",
        "num_non_empty_texts",
        "num_characters",
        "num_words",
        "whitespace_tokens",
        "vocab_size",
        "avg_words_per_text",
        "lexical_diversity",
        "truncated",
        "max_texts",
        "max_characters",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = result.to_dict()
            row.pop("top_words")
            writer.writerow(row)

    return output_path


def save_json(results: list[TrainWordCountMetrics], output_dir: Path) -> Path:
    output_path = output_dir / "train_word_counts.json"
    output_path.write_text(
        json.dumps([result.to_dict() for result in results], indent=2),
        encoding="utf-8",
    )
    return output_path


def plot_bar(
    results: list[TrainWordCountMetrics],
    metric_name: str,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    log_scale: bool = True,
) -> None:
    dataset_names = [result.dataset_name for result in results]
    values = [getattr(result, metric_name) for result in results]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#FFFFFF")
    bars = ax.bar(dataset_names, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if log_scale and any(value > 0 for value in values):
        ax.set_yscale("log")

    for bar, value in zip(bars, values):
        if value <= 0:
            continue
        label_y = value * (1.08 if log_scale else 1.01)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            compact_number(float(value), 0),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_plots(results: list[TrainWordCountMetrics], output_dir: Path) -> list[Path]:
    plot_paths = [
        output_dir / "train_total_words.png",
        output_dir / "train_vocab_size.png",
        output_dir / "train_whitespace_tokens.png",
    ]
    plot_bar(
        results,
        "num_words",
        "Train Word Count by Dataset",
        "Regex words in train split",
        plot_paths[0],
    )
    plot_bar(
        results,
        "vocab_size",
        "Train Vocabulary Size by Dataset",
        "Unique regex words in train split",
        plot_paths[1],
    )
    plot_bar(
        results,
        "whitespace_tokens",
        "Train Whitespace Token Count by Dataset",
        "Whitespace-delimited tokens in train split",
        plot_paths[2],
    )
    return plot_paths


def evaluate_train_word_counts(
    dataset_names: tuple[str, ...] = DEFAULT_DATASETS,
    *,
    max_texts: int | None = None,
    max_characters: int | None = None,
    top_k: int = 20,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[list[TrainWordCountMetrics], dict[str, Path | list[Path]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        count_train_words(
            dataset_name,
            max_texts=max_texts,
            max_characters=max_characters,
            top_k=top_k,
        )
        for dataset_name in dataset_names
    ]

    csv_path = save_csv(results, output_dir)
    json_path = save_json(results, output_dir)
    plot_paths = save_plots(results, output_dir)

    return results, {
        "csv": csv_path,
        "json": json_path,
        "plots": plot_paths,
    }
