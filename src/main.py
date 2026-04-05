import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.load_data import load

# EDA task
from src.eda.text8 import run_text8_eda
from src.eda.wikitext_103 import run_wikitext_103_eda


EDA_RUNNERS = {
    "text8": run_text8_eda,
    "wikitext-103": run_wikitext_103_eda,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset download, EDA, training, and evaluation tasks.")
    parser.add_argument("--download", type=str, help="Download and save a dataset to data/raw/<dataset_name>.")
    parser.add_argument("--eda", type=str, help="Run EDA for a dataset and save outputs to outputs/eda/<dataset_name>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.download:
        print(f"Downloading dataset: {args.download}")
        load(args.download)
        print(f"Saved dataset to data/raw/{args.download}")

    if args.eda:
        if args.eda not in EDA_RUNNERS:
            raise ValueError(f"Unsupported EDA dataset: {args.eda}")

        print(f"Running EDA for dataset: {args.eda}")
        EDA_RUNNERS[args.eda]()
        print(f"Saved EDA outputs to outputs/eda/{args.eda}")

    if not args.download and not args.eda:
        raise ValueError("No task selected. Use --download <dataset_name> or --eda <dataset_name>.")


if __name__ == "__main__":
    main()
