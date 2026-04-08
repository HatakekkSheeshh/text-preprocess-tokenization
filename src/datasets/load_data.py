import tarfile
import warnings
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from datasets import DatasetDict, load_dataset, load_from_disk  # type: ignore


BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ONE_BILLION_WORD_URL = "https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
ONE_BILLION_WORD_ARCHIVE_NAME = "one-billion-word.tar.gz"
ONE_BILLION_WORD_EXTRACT_DIR_NAME = "one-billion-word-source"
ONE_BILLION_WORD_ROOT_DIR_NAME = "1-billion-word-language-modeling-benchmark-r13output"
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


def load_enwik8() -> Path:
    path = get_dataset_path("enwik8")
    legacy_file_path = path
    dataset_dir = path.parent / "enwik8"
    file_path = dataset_dir / "enwik8"

    if file_path.exists():
        return file_path

    if legacy_file_path.exists() and legacy_file_path.is_file():
        return legacy_file_path

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_dir.parent / "enwik8.zip"

    urlretrieve(ENWIK8_URL, zip_path)

    with ZipFile(zip_path, "r") as zip_file:
        zip_file.extract("enwik8", dataset_dir)

    return file_path


def load_one_billion_word() -> DatasetDict:
    path = get_dataset_path("one-billion-word")
    if path.exists():
        return load_from_disk(str(path))

    path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = path.parent / ONE_BILLION_WORD_ARCHIVE_NAME
    extract_dir = path.parent / ONE_BILLION_WORD_EXTRACT_DIR_NAME

    if not archive_path.exists():
        urlretrieve(ONE_BILLION_WORD_URL, archive_path)

    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

    root_dir = extract_dir / ONE_BILLION_WORD_ROOT_DIR_NAME
    if not root_dir.exists():
        root_candidates = sorted(
            candidate
            for candidate in extract_dir.glob("1-billion-word-language-modeling-benchmark-r13output*")
            if candidate.is_dir()
        )
        if not root_candidates:
            raise FileNotFoundError(f"No extracted One Billion Word root directory found under: {extract_dir}")
        root_dir = root_candidates[0]

    train_dir = root_dir / "training-monolingual.tokenized.shuffled"
    heldout_dir = root_dir / "heldout-monolingual.tokenized.shuffled"

    train_files = sorted(str(file_path) for file_path in train_dir.glob("news.en-*") if file_path.is_file())
    heldout_files: list[str] = []
    if heldout_dir.exists():
        heldout_files = sorted(
            str(file_path) for file_path in heldout_dir.glob("news.en.heldout-*") if file_path.is_file()
        )

    if not train_files:
        raise FileNotFoundError(f"No train files found under: {train_dir}")

    if not heldout_files:
        if len(train_files) < 3:
            raise FileNotFoundError(
                f"No heldout files found under: {heldout_dir}, and not enough training shards to create fallback splits."
            )

        warnings.warn(
            (
                f"No heldout files found under: {heldout_dir}. "
                "Falling back to the last two training shards for validation/test."
            ),
            stacklevel=2,
        )
        heldout_files = train_files[-2:]
        train_files = train_files[:-2]

    ds = load_dataset(
        "text",
        data_files={
            "train": train_files,
            "validation": heldout_files,
            "test": heldout_files,
        },
    )
    ds.save_to_disk(str(path))
    return ds


def load(dataset_name: str) -> DatasetDict | Path:
    path = get_dataset_path(dataset_name)

    if dataset_name == "enwik8":
        return load_enwik8()

    if dataset_name == "one-billion-word":
        return load_one_billion_word()

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
