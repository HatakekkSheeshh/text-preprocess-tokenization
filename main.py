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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--eval", action="store_true", help="Evaluate tokenization on a dataset.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name for --eval.")
    parser.add_argument("--train-ngram", type=str, default=None, help="Train an n-gram language model on a dataset.")
    parser.add_argument("--tokenizer", type=str, default="word", choices=["word", "char", "bpe"])
    parser.add_argument("--ngram-order", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--laplace-alpha", type=float, default=1.0)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=50_000)
    parser.add_argument("--max-fit-texts", type=int, default=None)
    parser.add_argument("--max-eval-texts-per-split", type=int, default=None)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-validation-tokens", type=int, default=None)
    parser.add_argument("--max-test-tokens", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
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
        load_data(args.load)

    if args.eda is not None:
        dataset_name = args.eda.lower()

        if dataset_name not in EDA_RUNNERS:
            raise ValueError(f"Unsupported EDA dataset: {args.eda}")

        print(f"Running EDA for dataset: {args.eda}")
        EDA_RUNNERS[dataset_name]()
        print(f"Saved EDA outputs to outputs/eda/{dataset_name}")

    if args.eval:
        if args.dataset is None:
            raise ValueError("`--dataset` is required when using `--eval`.")

        result, saved_path = evaluate_and_save_tokenizer_on_dataset(
            dataset_name=args.dataset.lower(),
            tokenizer_name=args.tokenizer,
            max_fit_texts=args.max_fit_texts,
            max_eval_texts_per_split=args.max_eval_texts_per_split,
            min_freq=args.min_freq,
            max_vocab_size=args.max_vocab_size,
        )
        print(
            "Saved tokenization evaluation to "
            f"{saved_path} for dataset={result.dataset_name}, tokenizer={result.tokenizer_name}"
        )

    if args.train_ngram is not None:
        config = NGramTrainingConfig(
            dataset_name=args.train_ngram.lower(),
            tokenizer_name=args.tokenizer,
            order=args.ngram_order,
            alpha=args.laplace_alpha,
            min_freq=args.min_freq,
            max_vocab_size=args.max_vocab_size,
            max_fit_texts=args.max_fit_texts,
            max_train_tokens=args.max_train_tokens,
            max_validation_tokens=args.max_validation_tokens,
            max_test_tokens=args.max_test_tokens,
            run_name=args.run_name,
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
