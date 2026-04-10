import argparse

# Load data
from src.datasets.load_data import load

# EDA task
from src.eda.enwik8 import run_eda
from src.eda.one_billion_word import run_one_billion_word_eda
from src.eda.text8 import run_text8_eda
from src.eda.wikitext_103 import run_wikitext_103_eda
from src.training.train_lstm import LSTMTrainingConfig, train_lstm_language_model

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
    parser.add_argument("--train-lstm", type=str, default=None, help="Train an LSTM language model on a dataset.")
    parser.add_argument("--tokenizer", type=str, default="word", choices=["word", "char", "bpe"])
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=50_000)
    parser.add_argument("--max-fit-texts", type=int, default=None)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-validation-tokens", type=int, default=None)
    parser.add_argument("--max-test-tokens", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.load is not None:
        from src.datasets.load_data import load

        load(args.load)

    if args.eda is not None:
        dataset_name = args.eda.lower()

        if dataset_name not in EDA_RUNNERS:
            raise ValueError(f"Unsupported EDA dataset: {args.eda}")

        print(f"Running EDA for dataset: {args.eda}")
        EDA_RUNNERS[dataset_name]()
        print(f"Saved EDA outputs to outputs/eda/{dataset_name}")

    if args.train_lstm is not None:
        config = LSTMTrainingConfig(
            dataset_name=args.train_lstm.lower(),
            tokenizer_name=args.tokenizer,
            sequence_length=args.sequence_length,
            stride=args.stride,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            seed=args.seed,
            num_workers=args.num_workers,
            min_freq=args.min_freq,
            max_vocab_size=args.max_vocab_size,
            max_fit_texts=args.max_fit_texts,
            max_train_tokens=args.max_train_tokens,
            max_validation_tokens=args.max_validation_tokens,
            max_test_tokens=args.max_test_tokens,
            device=args.device,
            run_name=args.run_name,
            log_interval=args.log_interval,
        )
        train_lstm_language_model(config)

            
    if not args.load and not args.eda and not args.train_lstm:
        raise ValueError(
            "No task selected. Use --load <dataset_name>, --eda <dataset_name>, or --train-lstm <dataset_name>."
        )


if __name__ == "__main__":
    main()
