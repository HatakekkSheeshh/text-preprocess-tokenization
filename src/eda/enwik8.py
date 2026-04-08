import csv
import html
import json
import re
import string
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tqdm import tqdm


DATASET_NAME = "enwik8"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda" / DATASET_NAME
PLOTS_DIR = OUTPUT_DIR / "plots"
DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "figures" / DATASET_NAME

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
XML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
WIKI_BRACKET_PATTERN = re.compile(r"\[\[.*?\]\]")
WIKI_BRACE_PATTERN = re.compile(r"\{\{.*?\}\}")

PUNCTUATION_CHARS = set(string.punctuation)
TOP_K_WORDS = 20
PLOT_FACE_COLOR = "#FFFDF7"
PLOT_ACCENT = "#C95F44"
PLOT_ACCENT_ALT = "#2B5F75"
PLOT_NEUTRAL = "#D6C6A8"
PLOT_TEXT = "#2A221C"
PLOT_GRID = "#E7DCCD"


def resolve_dataset_file() -> Path:
    if DATASET_PATH.is_file():
        return DATASET_PATH

    nested_file = DATASET_PATH / "enwik8"
    if nested_file.is_file():
        return nested_file

    raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH} or {nested_file}")


def make_figure_directories() -> list[Path]:
    directories = [PLOTS_DIR, DOCS_FIGURES_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def save_figure(fig: plt.Figure, file_name: str) -> None:
    for directory in make_figure_directories():
        fig.savefig(directory / file_name, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_top_words(vocab_counter: Counter) -> None:
    top_words = vocab_counter.most_common(TOP_K_WORDS)
    if not top_words:
        return

    labels = [word for word, _ in reversed(top_words)]
    frequencies = [frequency for _, frequency in reversed(top_words)]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PLOT_FACE_COLOR)
    ax.barh(labels, frequencies, color=PLOT_ACCENT)
    ax.set_title("Enwik8 Top Words", fontsize=15, fontweight="bold", color=PLOT_TEXT)
    ax.set_xlabel("Frequency", color=PLOT_TEXT)
    ax.set_ylabel("Word", color=PLOT_TEXT)
    ax.set_facecolor(PLOT_FACE_COLOR)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, color=PLOT_GRID, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PLOT_GRID)
    ax.spines["bottom"].set_color(PLOT_GRID)
    ax.tick_params(colors=PLOT_TEXT)

    save_figure(fig, "top_words.png")


def plot_text_profile(summary: dict) -> None:
    profile = {
        "Vocabulary Size": summary["vocabulary_distribution"]["total_vocab_size"],
        "Avg Sentence Length": summary["sentence_length"]["average_length_in_words"],
        "Punctuation %": summary["punctuation_presence"]["punctuation_ratio_percent"],
        "Rare Word %": summary["naturalness_of_language"]["rare_word_ratio_percent"],
    }

    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=PLOT_FACE_COLOR)
    ax.bar(profile.keys(), profile.values(), color=[PLOT_ACCENT, PLOT_ACCENT_ALT, PLOT_NEUTRAL, PLOT_ACCENT])
    ax.set_title("Enwik8 Text Profile", fontsize=15, fontweight="bold", color=PLOT_TEXT)
    ax.set_ylabel("Value", color=PLOT_TEXT)
    ax.set_facecolor(PLOT_FACE_COLOR)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color=PLOT_GRID, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PLOT_GRID)
    ax.spines["bottom"].set_color(PLOT_GRID)
    ax.tick_params(colors=PLOT_TEXT, rotation=10)

    save_figure(fig, "text_profile.png")


def run_eda():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    dataset_file = resolve_dataset_file()

    with dataset_file.open("r", encoding="utf-8", errors="ignore") as file:
        raw_text = file.read()

    raw_text = html.unescape(raw_text)
    raw_text = html.unescape(raw_text)
    raw_text = raw_text.lower()

    vocab_counter = Counter()
    sentence_lengths = []

    tag_count = len(XML_TAG_PATTERN.findall(raw_text))
    url_count = len(URL_PATTERN.findall(raw_text))
    bracket_count = len(WIKI_BRACKET_PATTERN.findall(raw_text))
    brace_count = len(WIKI_BRACE_PATTERN.findall(raw_text))

    non_space_chars = [ch for ch in raw_text if not ch.isspace()]
    non_space_char_count = len(non_space_chars)
    punctuation_count = sum(ch in PUNCTUATION_CHARS for ch in non_space_chars)

    words = WORD_PATTERN.findall(raw_text)
    vocab_counter.update(words)

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(raw_text) if sentence.strip()]
    for sentence in tqdm(sentences, desc="Analyzing sentences", unit="sentence"):
        word_count = len(WORD_PATTERN.findall(sentence))
        if word_count > 0:
            sentence_lengths.append(word_count)

    avg_sentence_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    punct_ratio = (punctuation_count / non_space_char_count) * 100 if non_space_char_count > 0 else 0
    rare_words_count = sum(1 for count in vocab_counter.values() if count <= 2)
    rare_ratio = (rare_words_count / len(vocab_counter)) * 100 if vocab_counter else 0

    summary = {
        "vocabulary_distribution": {
            "total_words": sum(vocab_counter.values()),
            "total_vocab_size": len(vocab_counter),
        },
        "sentence_length": {
            "average_length_in_words": round(avg_sentence_len, 2),
            "note": "This metric is heavily skewed on raw data. Periods are overloaded in image extensions/URLs, and sentence boundaries are severely disrupted by Wiki Markdown.",
        },
        "punctuation_presence": {
            "punctuation_ratio_percent": round(punct_ratio, 2),
            "punctuation_count": punctuation_count,
            "note": "The punctuation count is heavily inflated by special characters used in MediaWiki syntax, specifically the abundant use of brackets [[ ]] for links and braces {{ }} for templates.",
        },
        "naturalness_of_language": {
            "total_xml_html_tags": tag_count,
            "tag_note": "This count encompasses both opening and closing tags.",
            "total_url_links": url_count,
            "url_note": "This count includes all URLs found in the text, regardless of their context or usage.",
            "total_wiki_brackets": bracket_count,
            "total_wiki_braces": brace_count,
            "rare_word_ratio_percent": round(rare_ratio, 2),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_filepath = OUTPUT_DIR / "top_50_most_frequent_words.csv"
    with csv_filepath.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["word", "frequency"])
        writer.writerows(vocab_counter.most_common(50))

    plot_top_words(vocab_counter)
    plot_text_profile(summary)

    return summary


if __name__ == "__main__":
    result = run_eda()
    print("\n--- ENWIK8 EDA RESULTS ---")
    print(json.dumps(result, indent=2))
