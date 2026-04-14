from __future__ import annotations

import argparse

import modal

from main import EDA_RUNNERS, SMOKE_LIMITS, expand_dataset_names
from src.datasets.load_data import load_data
from src.evaluation.tokenization import evaluate_and_save_tokenizer_on_dataset
from src.training.train_ngram import NGramTrainingConfig, train_ngram_language_model


APP_NAME = "text-preprocess-tokenization"
REMOTE_PROJECT_DIR = "/root/project"

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name(f"{APP_NAME}-data", create_if_missing=True)
outputs_volume = modal.Volume.from_name(f"{APP_NAME}-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .workdir(REMOTE_PROJECT_DIR)
    .add_local_dir(
        ".",
        remote_path=REMOTE_PROJECT_DIR,
        ignore=[
            ".git/**",
            "venv/**",
            "__pycache__/**",
            ".pytest_cache/**",
            "data/**",
            "outputs/**",
            "notebooks/**",
            "test/outputs/**",
        ],
    )
)


def _limit_value(
    dataset_name: str,
    option_name: str,
    value: int | None,
    smoke: bool,
) -> int | None:
    if value is not None or not smoke:
        return value
    if dataset_name not in SMOKE_LIMITS:
        supported_names = ", ".join(SMOKE_LIMITS)
        raise ValueError(f"Unsupported smoke dataset: {dataset_name}. Supported datasets: {supported_names}")
    return SMOKE_LIMITS[dataset_name][option_name]


def parse_args(argv: tuple[str, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default=None, help="Load a dataset, or use 'all'.")
    parser.add_argument("--eda", type=str, default=None, help="Run EDA for a dataset, or use 'all'.")
    parser.add_argument("--eval", action="store_true", help="Evaluate tokenization on a dataset.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name for --eval, or use 'all'.")
    parser.add_argument(
        "--train-ngram",
        type=str,
        default=None,
        help="Train an n-gram language model on a dataset, or use 'all'.",
    )
    parser.add_argument("--tokenizer", type=str, default="word", choices=["word", "char", "bpe"])
    parser.add_argument("--ngram-order", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--laplace-alpha", type=float, default=1.0)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=50_000)
    parser.add_argument("--max-fit-texts", type=int, default=None)
    parser.add_argument("--max-fit-characters", type=int, default=None)
    parser.add_argument("--max-eval-texts-per-split", type=int, default=None)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-train-characters", type=int, default=None)
    parser.add_argument("--max-validation-tokens", type=int, default=None)
    parser.add_argument("--max-validation-characters", type=int, default=None)
    parser.add_argument("--max-test-tokens", type=int, default=None)
    parser.add_argument("--max-test-characters", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--predict-context", action="append", default=[])
    parser.add_argument("--score-text", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    volumes={
        f"{REMOTE_PROJECT_DIR}/data": data_volume,
        f"{REMOTE_PROJECT_DIR}/outputs": outputs_volume,
    },
    timeout=60 * 60 * 6,
)
def load_dataset_modal(dataset_name: str) -> list[str]:
    loaded = []
    for name in expand_dataset_names(dataset_name):
        print(f"Loading dataset: {name}")
        load_data(name)
        loaded.append(name)

    data_volume.commit()
    return loaded


@app.function(
    image=image,
    volumes={
        f"{REMOTE_PROJECT_DIR}/data": data_volume,
        f"{REMOTE_PROJECT_DIR}/outputs": outputs_volume,
    },
    timeout=60 * 60 * 6,
)
def run_eda_modal(dataset_name: str) -> list[str]:
    completed = []
    for name in expand_dataset_names(dataset_name):
        if name not in EDA_RUNNERS:
            raise ValueError(f"Unsupported EDA dataset: {name}")

        print(f"Running EDA for dataset: {name}")
        EDA_RUNNERS[name]()
        completed.append(name)

    outputs_volume.commit()
    return completed


@app.function(
    image=image,
    volumes={
        f"{REMOTE_PROJECT_DIR}/data": data_volume,
        f"{REMOTE_PROJECT_DIR}/outputs": outputs_volume,
    },
    timeout=60 * 60 * 6,
)
def evaluate_tokenizer_modal(
    dataset_name: str,
    tokenizer_name: str = "word",
    min_freq: int = 1,
    max_vocab_size: int | None = 50_000,
    max_fit_texts: int | None = None,
    max_eval_texts_per_split: int | None = None,
    smoke: bool = False,
) -> list[dict]:
    results = []
    for name in expand_dataset_names(dataset_name):
        result, saved_path = evaluate_and_save_tokenizer_on_dataset(
            dataset_name=name,
            tokenizer_name=tokenizer_name,
            max_fit_texts=_limit_value(name, "max_fit_texts", max_fit_texts, smoke),
            max_eval_texts_per_split=_limit_value(
                name,
                "max_eval_texts_per_split",
                max_eval_texts_per_split,
                smoke,
            ),
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
        )
        results.append(
            {
                "dataset_name": result.dataset_name,
                "tokenizer_name": result.tokenizer_name,
                "saved_path": str(saved_path),
            }
        )

    outputs_volume.commit()
    return results


@app.function(
    image=image,
    volumes={
        f"{REMOTE_PROJECT_DIR}/data": data_volume,
        f"{REMOTE_PROJECT_DIR}/outputs": outputs_volume,
    },
    timeout=60 * 60 * 6,
)
def train_ngram_modal(
    dataset_name: str,
    tokenizer_name: str = "word",
    order: int = 3,
    alpha: float = 1.0,
    min_freq: int = 1,
    max_vocab_size: int | None = 50_000,
    max_fit_texts: int | None = None,
    max_fit_characters: int | None = None,
    max_train_tokens: int | None = None,
    max_train_characters: int | None = None,
    max_validation_tokens: int | None = None,
    max_validation_characters: int | None = None,
    max_test_tokens: int | None = None,
    max_test_characters: int | None = None,
    run_name: str | None = None,
    smoke: bool = False,
    prediction_contexts: list[str] | None = None,
    score_texts: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    summaries = []
    dataset_names = expand_dataset_names(dataset_name)

    for name in dataset_names:
        current_run_name = run_name
        if run_name is not None and len(dataset_names) > 1:
            current_run_name = f"{run_name}_{name}"

        config = NGramTrainingConfig(
            dataset_name=name,
            tokenizer_name=tokenizer_name,
            order=order,
            alpha=alpha,
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
            max_fit_texts=_limit_value(name, "max_fit_texts", max_fit_texts, smoke),
            max_fit_characters=_limit_value(name, "max_fit_characters", max_fit_characters, smoke),
            max_train_tokens=_limit_value(name, "max_train_tokens", max_train_tokens, smoke),
            max_train_characters=_limit_value(
                name,
                "max_train_characters",
                max_train_characters,
                smoke,
            ),
            max_validation_tokens=_limit_value(
                name,
                "max_validation_tokens",
                max_validation_tokens,
                smoke,
            ),
            max_validation_characters=_limit_value(
                name,
                "max_validation_characters",
                max_validation_characters,
                smoke,
            ),
            max_test_tokens=_limit_value(name, "max_test_tokens", max_test_tokens, smoke),
            max_test_characters=_limit_value(
                name,
                "max_test_characters",
                max_test_characters,
                smoke,
            ),
            run_name=current_run_name,
        )
        summaries.append(
            train_ngram_language_model(
                config,
                prediction_contexts=prediction_contexts,
                score_texts=score_texts,
                top_k=top_k,
            )
        )

    data_volume.commit()
    outputs_volume.commit()
    return summaries


@app.local_entrypoint()
def main(*argv: str):
    args = parse_args(argv)

    if args.load is not None:
        loaded = load_dataset_modal.remote(args.load)
        print(f"Loaded datasets: {', '.join(loaded)}")

    if args.eda is not None:
        completed = run_eda_modal.remote(args.eda)
        print(f"Completed EDA for datasets: {', '.join(completed)}")

    if args.eval:
        if args.dataset is None:
            raise ValueError("`--dataset` is required when using `--eval`.")

        results = evaluate_tokenizer_modal.remote(
            dataset_name=args.dataset,
            tokenizer_name=args.tokenizer,
            min_freq=args.min_freq,
            max_vocab_size=args.max_vocab_size,
            max_fit_texts=args.max_fit_texts,
            max_eval_texts_per_split=args.max_eval_texts_per_split,
            smoke=args.smoke,
        )
        for result in results:
            print(
                "Saved tokenization evaluation to "
                f"{result['saved_path']} for dataset={result['dataset_name']}, "
                f"tokenizer={result['tokenizer_name']}"
            )

    if args.train_ngram is not None:
        train_ngram_modal.remote(
            dataset_name=args.train_ngram,
            tokenizer_name=args.tokenizer,
            order=args.ngram_order,
            alpha=args.laplace_alpha,
            min_freq=args.min_freq,
            max_vocab_size=args.max_vocab_size,
            max_fit_texts=args.max_fit_texts,
            max_fit_characters=args.max_fit_characters,
            max_train_tokens=args.max_train_tokens,
            max_train_characters=args.max_train_characters,
            max_validation_tokens=args.max_validation_tokens,
            max_validation_characters=args.max_validation_characters,
            max_test_tokens=args.max_test_tokens,
            max_test_characters=args.max_test_characters,
            run_name=args.run_name,
            smoke=args.smoke,
            prediction_contexts=args.predict_context,
            score_texts=args.score_text,
            top_k=args.top_k,
        )

    if not args.load and not args.eda and not args.eval and not args.train_ngram:
        raise ValueError(
            "No task selected. Use --load <dataset_name>, --eda <dataset_name>, "
            "--eval --dataset <dataset_name>, or --train-ngram <dataset_name>."
        )
