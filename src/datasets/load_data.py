from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk  # type: ignore


BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASET_SOURCES = {
    "wikitext-103": ("Salesforce/wikitext", "wikitext-103-v1"),
    "text8": ("afmck/text8", None),
}


def get_dataset_path(dataset_name: str) -> Path:
    return BASE_DIR / dataset_name


def save(dataset_name: str, ds: DatasetDict) -> None:
    path = get_dataset_path(dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))


def load(dataset_name: str) -> DatasetDict:
    path = get_dataset_path(dataset_name)
    if path.exists():
        return load_from_disk(str(path))

    if dataset_name not in DATASET_SOURCES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_path, dataset_config = DATASET_SOURCES[dataset_name]
    if dataset_config is None:
        ds = load_dataset(dataset_path)
    else:
        ds = load_dataset(dataset_path, dataset_config)

    save(dataset_name, ds)
    return ds
