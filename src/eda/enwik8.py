import json
import re
import csv
import string
from collections import Counter
from pathlib import Path

DATASET_NAME = "enwik8"
CURRENT_DIR = Path(__file__).resolve().parent
DATASET_PATH = CURRENT_DIR / "data" / "raw" /DATASET_NAME
OUTPUT_DIR = CURRENT_DIR / "outputs" / "eda" / DATASET_NAME

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
XML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
WIKI_BRACKET_PATTERN = re.compile(r"\[\[.*?\]\]")
WIKI_BRACE_PATTERN = re.compile(r"\{\{.*?\}\}")

PUNCTUATION_CHARS = set(string.punctuation)

def run_eda():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")
    
    with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    vocab_counter = Counter()
    sentence_lengths = []

    # Counting XML/HTML tags, URLs, and Wiki Markdown syntax...
    tag_count = len(XML_TAG_PATTERN.findall(raw_text))
    url_count = len(URL_PATTERN.findall(raw_text))
    bracket_count = len(WIKI_BRACKET_PATTERN.findall(raw_text))
    brace_count = len(WIKI_BRACE_PATTERN.findall(raw_text))

    #  Analyzing punctuation presence...
    non_space_chars = [ch for ch in raw_text if not ch.isspace()]
    non_space_char_count = len(non_space_chars)
    punctuation_count = sum(ch in PUNCTUATION_CHARS for ch in non_space_chars)

    # Tokenizing and analyzing vocabulary distribution...
    words = WORD_PATTERN.findall(raw_text.lower())
    vocab_counter.update(words)

    # Splitting sentences and calculating lengths...
    sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(raw_text) if s.strip()]
    for sentence in sentences:
        word_count = len(WORD_PATTERN.findall(sentence))
        if word_count > 0:
            sentence_lengths.append(word_count)

    # Generating JSON summary and CSV file...
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
            "rare_word_ratio_percent": round(rare_ratio, 2)
        }
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top_k = 50
    csv_filename = f"top_{top_k}_most_frequent_words.csv"
    csv_filepath = OUTPUT_DIR / csv_filename

    with csv_filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"]) # Tiêu đề cột
        writer.writerows(vocab_counter.most_common(top_k)) # Ghi dữ liệu
    
    return summary

if __name__ == "__main__":
    result = run_eda()
    print("\n--- ENWIK8 EDA RESULTS ---")
    print(json.dumps(result, indent=2))