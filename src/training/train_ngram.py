from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time

from src.datasets.load_data import load_text_dataset
from src.models.ngram import NGramLanguageModel
from src.tokenizers import BaseTokenizer, build_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "artifacts" / "ngram"
METRICS_ROOT = PROJECT_ROOT / "outputs" / "metrics" / "ngram"
REQUIRED_SPLITS = ("train", "validation", "test")


@dataclass
class NGramTrainingConfig:
    dataset_name: str
    tokenizer_name: str = "word"
    order: int = 3
    alpha: float = 1.0
    min_freq: int = 1
    max_vocab_size: int | None = 50_000
    max_fit_texts: int | None = None
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
) -> list[int]:
    return tokenizer.encode_texts(text_dataset.iter_texts(split_name), max_tokens=max_tokens)


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
    return {
        "text": text,
        "tokens": tokenizer.decode_ids(token_ids),
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
    tokenizer.fit_from_texts(limit_texts(text_dataset.iter_texts("train"), config.max_fit_texts))
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
    model.fit(encoded_splits["train"])
    model_fit_seconds = time.perf_counter() - model_fit_start

    run_name = build_run_name(config)
    artifact_dir = ARTIFACT_ROOT / run_name
    metrics_path = METRICS_ROOT / f"{run_name}.json"
    tokenizer_path = artifact_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    split_scores = {
        split_name: model.score_sequence(token_ids).to_dict()
        for split_name, token_ids in encoded_splits.items()
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
                "num_tokens": len(token_ids),
            }
            for split_name, token_ids in encoded_splits.items()
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
        f"val ppl {split_scores['validation']['perplexity']:.4f} | "
        f"test ppl {split_scores['test']['perplexity']:.4f}"
    )

    for payload in predictions:
        print(f"Top {top_k} predictions for context: {payload['context_text']!r}")
        for item in payload["predictions"]:
            print(f"  {item['token']!r}: {item['probability']:.6f}")

    for payload in scored_texts:
        print(
            f"Score for text {payload['text']!r}: "
            f"log_prob={payload['log_probability']:.4f}, "
            f"ppl={payload['perplexity']:.4f}"
        )

    print(f"Saved artifacts to {artifact_dir}")
    print(f"Saved metrics to {metrics_path}")
    return summary
