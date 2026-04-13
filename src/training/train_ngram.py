from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time

from src.datasets.load_data import load_text_dataset
from src.models.ngram import NGramLanguageModel
from src.tokenizers import BaseTokenizer, build_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "artifacts" / "ngram"
METRICS_ROOT = PROJECT_ROOT / "outputs" / "metrics" / "ngram"
REQUIRED_SPLITS = ("train", "validation", "test")
LN_2 = math.log(2.0)


@dataclass
class EncodedSplit:
    token_ids: list[int]
    num_characters: int


@dataclass
class NGramTrainingConfig:
    dataset_name: str
    tokenizer_name: str = "word"
    order: int = 3
    alpha: float = 1.0
    min_freq: int = 1
    max_vocab_size: int | None = 50_000
    max_fit_texts: int | None = None
    max_fit_characters: int | None = None
    max_train_tokens: int | None = None
    max_validation_tokens: int | None = None
    max_test_tokens: int | None = None
    run_name: str | None = None


def build_run_name(config: NGramTrainingConfig) -> str:
    if config.run_name:
        return config.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ngram_{config.dataset_name}_{config.tokenizer_name}_{config.order}gram_{timestamp}"


def limit_texts(texts, max_texts: int | None):
    if max_texts is None:
        yield from texts
        return

    for index, text in enumerate(texts):
        if index >= max_texts:
            break
        yield text


def limit_texts_for_fit(
    texts,
    *,
    max_texts: int | None = None,
    max_characters: int | None = None,
):
    yielded_texts = 0
    consumed_characters = 0

    for text in texts:
        if max_texts is not None and yielded_texts >= max_texts:
            break

        if max_characters is None:
            yield text
            yielded_texts += 1
            continue

        remaining_characters = max_characters - consumed_characters
        if remaining_characters <= 0:
            break

        if len(text) <= remaining_characters:
            yield text
            yielded_texts += 1
            consumed_characters += len(text)
            continue

        # For single-stream corpora such as text8/enwik8, slicing the first text
        # still gives us a representative small-fit subset for smoke/Colab runs.
        yield text[:remaining_characters]
        break


def validate_required_splits(dataset_name: str, split_names: tuple[str, ...]) -> None:
    missing_splits = [split_name for split_name in REQUIRED_SPLITS if split_name not in split_names]
    if missing_splits:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required splits: {', '.join(missing_splits)}. "
            f"Available splits: {', '.join(split_names)}"
        )


def encode_split(
    text_dataset,
    tokenizer: BaseTokenizer,
    split_name: str,
    *,
    max_tokens: int | None = None,
) -> EncodedSplit:
    token_ids: list[int] = []
    num_characters = 0
    first_non_empty = True
    boundary_token_ids = tokenizer.encode_tokens(tokenizer.boundary_tokens())

    for text in text_dataset.iter_texts(split_name):
        if not text:
            continue

        encoded_text = tokenizer.encode_text(text)
        if not encoded_text:
            continue

        if not first_non_empty:
            if max_tokens is not None:
                remaining = max_tokens - len(token_ids)
                if remaining <= 0:
                    break
                if remaining < len(boundary_token_ids):
                    token_ids.extend(boundary_token_ids[:remaining])
                    break

            token_ids.extend(boundary_token_ids)

        if max_tokens is None or len(token_ids) + len(encoded_text) <= max_tokens:
            token_ids.extend(encoded_text)
            num_characters += len(text)
            first_non_empty = False
            continue

        remaining = max_tokens - len(token_ids)
        if remaining <= 0:
            break

        token_ids.extend(encoded_text[:remaining])
        num_characters += tokenizer.count_characters_for_token_prefix(text, remaining)
        break

    return EncodedSplit(token_ids=token_ids, num_characters=num_characters)


def compute_bits_per_character(*, log_probability: float, num_characters: int) -> float:
    if num_characters <= 0:
        return 0.0
    return -log_probability / (num_characters * LN_2)


def build_prediction_payload(
    model: NGramLanguageModel,
    tokenizer: BaseTokenizer,
    context_text: str,
    *,
    top_k: int,
) -> dict:
    context_ids = tokenizer.encode_texts([context_text])
    predictions = model.predict_next(
        context_ids,
        top_k=top_k,
        excluded_token_ids=[tokenizer.pad_token_id, tokenizer.unk_token_id],
    )

    return {
        "context_text": context_text,
        "context_tokens": tokenizer.decode_ids(context_ids),
        "predictions": [
            {
                "token_id": item.token_id,
                "token": tokenizer.decode_ids([item.token_id])[0],
                "probability": item.probability,
            }
            for item in predictions
        ],
    }


def build_score_payload(
    model: NGramLanguageModel,
    tokenizer: BaseTokenizer,
    text: str,
) -> dict:
    token_ids = tokenizer.encode_texts([text])
    score = model.score_sequence(token_ids)
    num_characters = len(text)
    return {
        "text": text,
        "tokens": tokenizer.decode_ids(token_ids),
        "num_characters": num_characters,
        "bits_per_character": compute_bits_per_character(
            log_probability=score.log_probability,
            num_characters=num_characters,
        ),
        **score.to_dict(),
    }


def train_ngram_language_model(
    config: NGramTrainingConfig,
    *,
    prediction_contexts: list[str] | None = None,
    score_texts: list[str] | None = None,
    top_k: int = 5,
) -> dict:
    if config.order not in {1, 2, 3}:
        raise ValueError("Supported n-gram orders are 1, 2, and 3.")

    text_dataset = load_text_dataset(config.dataset_name)
    validate_required_splits(config.dataset_name, text_dataset.split_names)

    tokenizer = build_tokenizer(
        config.tokenizer_name,
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
    )

    fit_start = time.perf_counter()
    tokenizer.fit_from_texts(
        limit_texts_for_fit(
            text_dataset.iter_texts("train"),
            max_texts=config.max_fit_texts,
            max_characters=config.max_fit_characters,
        )
    )
    tokenizer_fit_seconds = time.perf_counter() - fit_start

    encoded_splits = {
        "train": encode_split(text_dataset, tokenizer, "train", max_tokens=config.max_train_tokens),
        "validation": encode_split(text_dataset, tokenizer, "validation", max_tokens=config.max_validation_tokens),
        "test": encode_split(text_dataset, tokenizer, "test", max_tokens=config.max_test_tokens),
    }

    model = NGramLanguageModel(
        order=config.order,
        vocab_size=tokenizer.vocab_size,
        alpha=config.alpha,
    )

    model_fit_start = time.perf_counter()
    model.fit(encoded_splits["train"].token_ids)
    model_fit_seconds = time.perf_counter() - model_fit_start

    run_name = build_run_name(config)
    artifact_dir = ARTIFACT_ROOT / run_name
    metrics_path = METRICS_ROOT / f"{run_name}.json"
    tokenizer_path = artifact_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    split_scores: dict[str, dict] = {}
    for split_name, encoded_split in encoded_splits.items():
        score = model.score_sequence(encoded_split.token_ids)
        split_scores[split_name] = {
            **score.to_dict(),
            "num_characters": encoded_split.num_characters,
            "bits_per_character": compute_bits_per_character(
                log_probability=score.log_probability,
                num_characters=encoded_split.num_characters,
            ),
        }

    predictions = [
        build_prediction_payload(model, tokenizer, context_text, top_k=top_k)
        for context_text in (prediction_contexts or [])
    ]
    scored_texts = [
        build_score_payload(model, tokenizer, text)
        for text in (score_texts or [])
    ]

    summary = {
        "run_name": run_name,
        "config": asdict(config),
        "tokenizer": {
            "type": tokenizer.tokenizer_type,
            "vocab_size": tokenizer.vocab_size,
        },
        "model": {
            "type": "ngram-language-model",
            "order": config.order,
            "alpha": config.alpha,
            "num_observed_contexts": model.num_observed_contexts,
            "num_observed_full_order_ngrams": model.num_observed_full_order_ngrams,
        },
        "timing": {
            "tokenizer_fit_seconds": tokenizer_fit_seconds,
            "model_fit_seconds": model_fit_seconds,
            "total_seconds": tokenizer_fit_seconds + model_fit_seconds,
        },
        "dataset": {
            split_name: {
                "num_tokens": len(encoded_split.token_ids),
                "num_characters": encoded_split.num_characters,
            }
            for split_name, encoded_split in encoded_splits.items()
        },
        "splits": split_scores,
        "prediction_contexts": predictions,
        "scored_texts": scored_texts,
        "artifact_dir": str(artifact_dir),
        "tokenizer_path": str(tokenizer_path),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Prepared n-gram run with "
        f"dataset={config.dataset_name}, tokenizer={config.tokenizer_name}, "
        f"order={config.order}, vocab_size={tokenizer.vocab_size}"
    )
    print(
        f"  train ppl {split_scores['train']['perplexity']:.4f} | "
        f"train bpc {split_scores['train']['bits_per_character']:.4f} | "
        f"val ppl {split_scores['validation']['perplexity']:.4f} | "
        f"val bpc {split_scores['validation']['bits_per_character']:.4f} | "
        f"test ppl {split_scores['test']['perplexity']:.4f} | "
        f"test bpc {split_scores['test']['bits_per_character']:.4f}"
    )

    for payload in predictions:
        print(f"Top {top_k} predictions for context: {payload['context_text']!r}")
        for item in payload["predictions"]:
            print(f"  {item['token']!r}: {item['probability']:.6f}")

    for payload in scored_texts:
        print(
            f"Score for text {payload['text']!r}: "
            f"log_prob={payload['log_probability']:.4f}, "
            f"bpc={payload['bits_per_character']:.4f}, "
            f"ppl={payload['perplexity']:.4f}"
        )

    print(f"Saved artifacts to {artifact_dir}")
    print(f"Saved metrics to {metrics_path}")
    return summary
