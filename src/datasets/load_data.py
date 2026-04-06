from datasets import Dataset, load_dataset, DatasetDict # type: ignore
import os
from pathlib import Path

BASE_DIR = os.path.join(Path(__file__).resolve().parents[2], "data/raw")

def save(dataset_name: str, ds: DatasetDict):
    path = os.path.join(BASE_DIR, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    ds.save_to_disk(path)

def load(dataset_name: str) -> DatasetDict:
    if dataset_name == "wikitext-103":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    elif dataset_name == "":
        pass
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    save(dataset_name, ds)
    return ds