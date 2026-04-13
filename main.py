import argparse

from src.datasets.load_data import load_data
from src.eda.enwik8 import run_eda
from src.eda.one_billion_word import run_one_billion_word_eda
from src.eda.text8 import run_text8_eda
from src.eda.wikitext_103 import run_wikitext_103_eda
from src.evaluation.tokenization import evaluate_and_save_tokenizer_on_dataset
from src.training.train_ngram import NGramTrainingConfig, train_ngram_language_model


EDA_RUNNERS = {
    "text8": run_text8_eda,
    "wikitext-103": run_wikitext_103_eda,
    "enwik8": run_eda,
    "one-billion-word": run_one_billion_word_eda,
}
SUPPORTED_DATASETS = tuple(EDA_RUNNERS.keys())
SMOKE_LIMITS = {
    "text8": {
        "max_fit_texts": 1_000,
        "max_fit_characters": 200_000,
        "max_eval_texts_per_split": 1_000,
        "max_train_tokens": 4_096,
        "max_validation_tokens": 1_024,
        "max_test_tokens": 1_024,
    },
    "wikitext-103": {
        "max_fit_texts": 1_000,
        "max_fit_characters": None,
        "max_eval_texts_per_split": 500,
        "max_train_tokens": 10_000,
        "max_validation_tokens": 2_000,
        "max_test_tokens": 2_000,
    },
    "enwik8": {
        "max_fit_texts": 1,
        "max_fit_characters": 200_000,
        "max_eval_texts_per_split": 1,
        "max_train_tokens": 20_000,
        "max_validation_tokens": 5_000,
        "max_test_tokens": 5_000,
    },
    "one-billion-word": {
        "max_fit_texts": 1_000,
        "max_fit_characters": None,
        "max_eval_texts_per_split": 500,
        "max_train_tokens": 20_000,
        "max_validation_tokens": 5_000,
        "max_test_tokens": 5_000,
    },
}


def expand_dataset_names(dataset_name: str) -> tuple[str, ...]:
    normalized_name = dataset_name.lower()
    if normalized_name == "all":
        return SUPPORTED_DATASETS
    return (normalized_name,)


def get_limit_value(args, dataset_name: str, option_name: str):
    value = getattr(args, option_name)
    if value is not None or not args.smoke:
        return value
    if dataset_name not in SMOKE_LIMITS:
        supported_names = ", ".join(SMOKE_LIMITS)
        raise ValueError(f"Unsupported smoke dataset: {dataset_name}. Supported datasets: {supported_names}")
    return SMOKE_LIMITS[dataset_name][option_name]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda", type=str, default=None, help="Run EDA for a dataset, or use 'all'.")
    parser.add_argument("--load", type=str, default=None, help="Load a dataset, or use 'all'.")
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
    parser.add_argument("--max-validation-tokens", type=int, default=None)
    parser.add_argument("--max-test-tokens", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use small dataset-specific limits for a quick smoke test.",
    )
    parser.add_argument(
        "--predict-context",
        action="append",
        default=[],
        help="Context string for next-token prediction. Can be passed multiple times.",
    )
    parser.add_argument(
        "--score-text",
        action="append",
        default=[],
        help="Text string to score with the fitted n-gram model. Can be passed multiple times.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.load is not None:
        for dataset_name in expand_dataset_names(args.load):
            load_data(dataset_name)

    if args.eda is not None:
        for dataset_name in expand_dataset_names(args.eda):
            if dataset_name not in EDA_RUNNERS:
                raise ValueError(f"Unsupported EDA dataset: {dataset_name}")

            print(f"Running EDA for dataset: {dataset_name}")
            EDA_RUNNERS[dataset_name]()
            print(f"Saved EDA outputs to outputs/eda/{dataset_name}")

    if args.eval:
        if args.dataset is None:
            raise ValueError("`--dataset` is required when using `--eval`.")

        for dataset_name in expand_dataset_names(args.dataset):
            result, saved_path = evaluate_and_save_tokenizer_on_dataset(
                dataset_name=dataset_name,
                tokenizer_name=args.tokenizer,
                max_fit_texts=get_limit_value(args, dataset_name, "max_fit_texts"),
                max_eval_texts_per_split=get_limit_value(args, dataset_name, "max_eval_texts_per_split"),
                min_freq=args.min_freq,
                max_vocab_size=args.max_vocab_size,
            )
            print(
                "Saved tokenization evaluation to "
                f"{saved_path} for dataset={result.dataset_name}, tokenizer={result.tokenizer_name}"
            )

    if args.train_ngram is not None:
        train_dataset_names = expand_dataset_names(args.train_ngram)
        for dataset_name in train_dataset_names:
            run_name = args.run_name
            if args.run_name is not None and len(train_dataset_names) > 1:
                run_name = f"{args.run_name}_{dataset_name}"

            config = NGramTrainingConfig(
                dataset_name=dataset_name,
                tokenizer_name=args.tokenizer,
                order=args.ngram_order,
                alpha=args.laplace_alpha,
                min_freq=args.min_freq,
                max_vocab_size=args.max_vocab_size,
                max_fit_texts=get_limit_value(args, dataset_name, "max_fit_texts"),
                max_fit_characters=get_limit_value(args, dataset_name, "max_fit_characters"),
                max_train_tokens=get_limit_value(args, dataset_name, "max_train_tokens"),
                max_validation_tokens=get_limit_value(args, dataset_name, "max_validation_tokens"),
                max_test_tokens=get_limit_value(args, dataset_name, "max_test_tokens"),
                run_name=run_name,
            )
            train_ngram_language_model(
                config,
                prediction_contexts=args.predict_context,
                score_texts=args.score_text,
                top_k=args.top_k,
            )

    if not args.load and not args.eda and not args.eval and not args.train_ngram:
        raise ValueError(
            "No task selected. Use --load <dataset_name>, --eda <dataset_name>, "
            "--eval --dataset <dataset_name>, or --train-ngram <dataset_name>."
        )


if __name__ == "__main__":
    main()
