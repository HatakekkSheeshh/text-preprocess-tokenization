import argparse

# Load data
from src.datasets.load_data import load

# EDA task
from src.eda.enwik8 import run_eda
from src.eda.text8 import run_text8_eda
from src.eda.wikitext_103 import run_wikitext_103_eda

EDA_RUNNERS = {
    "text8": run_text8_eda,
    "wikitext-103": run_wikitext_103_eda,
    "enwik8": run_eda,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
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

            
    if not args.load and not args.eda:
        raise ValueError("No task selected. Use --load <dataset_name> or --eda <dataset_name>.")


if __name__ == "__main__":
    main()
