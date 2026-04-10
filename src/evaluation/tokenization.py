from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

from src.datasets.load_data import load_text_dataset
from src.tokenizers import build_tokenizer


REQUIRED_SPLITS = ("train", "validation", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZATION_METRICS_ROOT = PROJECT_ROOT / "outputs" / "metrics" / "tokenization"


@dataclass
class TokenizationSplitMetrics:
    num_texts: int
    num_characters: int
    num_tokens: int
    avg_tokens_per_text: float
    median_tokens_per_text: float
    p95_tokens_per_text: float
    max_tokens_per_text: int
    unk_count: int
    unk_ratio: float
    encode_seconds: float
    tokens_per_second: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TokenizationEvaluationResult:
    dataset_name: str
    tokenizer_name: str
    vocab_size: int
    fit_seconds: float
    splits: dict[str, TokenizationSplitMetrics]

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "tokenizer_name": self.tokenizer_name,
            "vocab_size": self.vocab_size,
            "fit_seconds": self.fit_seconds,
            "splits": {
                split_name: split_metrics.to_dict()
                for split_name, split_metrics in self.splits.items()
            },
        }


def limit_texts(texts, max_texts: int | None):
    if max_texts is None:
        yield from texts
        return

    for index, text in enumerate(texts):
        if index >= max_texts:
            break
        yield text


def compute_percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)

    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = rank - lower_index
    return float(lower_value + (upper_value - lower_value) * weight)


def compute_median(values: list[int]) -> float:
    return compute_percentile(values, 0.5)


def validate_required_splits(dataset_name: str, split_names: tuple[str, ...]) -> None:
    missing_splits = [split_name for split_name in REQUIRED_SPLITS if split_name not in split_names]
    if missing_splits:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required splits: {', '.join(missing_splits)}. "
            f"Available splits: {', '.join(split_names)}"
        )


def collect_texts(text_dataset, split_name: str, max_texts: int | None) -> list[str]:
    return list(limit_texts(text_dataset.iter_texts(split_name), max_texts))


def evaluate_split(tokenizer, texts: list[str]) -> TokenizationSplitMetrics:
    token_lengths: list[int] = []
    num_characters = sum(len(text) for text in texts)
    total_tokens = 0
    unk_count = 0

    encode_start = time.perf_counter()
    for text in texts:
        token_ids = tokenizer.encode_texts([text])
        token_lengths.append(len(token_ids))
        total_tokens += len(token_ids)
        unk_count += sum(1 for token_id in token_ids if token_id == tokenizer.unk_token_id)
    encode_seconds = time.perf_counter() - encode_start

    num_texts = len(texts)
    avg_tokens_per_text = (total_tokens / num_texts) if num_texts else 0.0
    unk_ratio = (unk_count / total_tokens) if total_tokens else 0.0
    tokens_per_second = (total_tokens / encode_seconds) if encode_seconds > 0 else 0.0

    return TokenizationSplitMetrics(
        num_texts=num_texts,
        num_characters=num_characters,
        num_tokens=total_tokens,
        avg_tokens_per_text=avg_tokens_per_text,
        median_tokens_per_text=compute_median(token_lengths),
        p95_tokens_per_text=compute_percentile(token_lengths, 0.95),
        max_tokens_per_text=max(token_lengths, default=0),
        unk_count=unk_count,
        unk_ratio=unk_ratio,
        encode_seconds=encode_seconds,
        tokens_per_second=tokens_per_second,
    )


def evaluate_tokenizer_on_dataset(
    dataset_name: str,
    tokenizer_name: str,
    *,
    max_fit_texts: int | None = None,
    max_eval_texts_per_split: int | None = None,
    min_freq: int = 1,
    max_vocab_size: int | None = 50_000,
) -> TokenizationEvaluationResult:
    text_dataset = load_text_dataset(dataset_name)
    validate_required_splits(dataset_name, text_dataset.split_names)

    tokenizer = build_tokenizer(
        tokenizer_name,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )

    train_texts_for_fit = collect_texts(text_dataset, "train", max_fit_texts)
    fit_start = time.perf_counter()
    tokenizer.fit_from_texts(train_texts_for_fit)
    fit_seconds = time.perf_counter() - fit_start

    split_metrics: dict[str, TokenizationSplitMetrics] = {}
    for split_name in REQUIRED_SPLITS:
        split_texts = collect_texts(text_dataset, split_name, max_eval_texts_per_split)
        split_metrics[split_name] = evaluate_split(tokenizer, split_texts)

    return TokenizationEvaluationResult(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        vocab_size=tokenizer.vocab_size,
        fit_seconds=fit_seconds,
        splits=split_metrics,
    )


def build_tokenization_metrics_path(
    dataset_name: str,
    tokenizer_name: str,
    *,
    output_dir: Path | None = None,
) -> Path:
    metrics_root = output_dir or TOKENIZATION_METRICS_ROOT
    return metrics_root / f"{dataset_name}_{tokenizer_name}.json"


def save_tokenization_evaluation(
    result: TokenizationEvaluationResult,
    *,
    output_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    path = output_path or build_tokenization_metrics_path(
        result.dataset_name,
        result.tokenizer_name,
        output_dir=output_dir,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def evaluate_and_save_tokenizer_on_dataset(
    dataset_name: str,
    tokenizer_name: str,
    *,
    max_fit_texts: int | None = None,
    max_eval_texts_per_split: int | None = None,
    min_freq: int = 1,
    max_vocab_size: int | None = 50_000,
    output_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[TokenizationEvaluationResult, Path]:
    result = evaluate_tokenizer_on_dataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_fit_texts=max_fit_texts,
        max_eval_texts_per_split=max_eval_texts_per_split,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )
    saved_path = save_tokenization_evaluation(
        result,
        output_path=output_path,
        output_dir=output_dir,
    )
    return result, saved_path
