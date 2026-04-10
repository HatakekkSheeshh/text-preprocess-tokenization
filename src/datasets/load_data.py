from __future__ import annotations

import tarfile
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
DEFAULT_TEXT_SPLITS = ("train", "validation", "test")
ENWIK8_SPLIT_RANGES = {
    "train": (0, 90_000_000),
    "validation": (90_000_000, 95_000_000),
    "test": (95_000_000, 100_000_000),
}


def get_dataset_path(dataset_name: str) -> Path:
    return BASE_DIR / dataset_name


def save_dataset(dataset_name: str, dataset: DatasetDict) -> None:
    path = get_dataset_path(dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))


class DatasetLoader(ABC):
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    @property
    def dataset_path(self) -> Path:
        return get_dataset_path(self.dataset_name)

    @abstractmethod
    def load(self) -> DatasetDict | Path:
        raise NotImplementedError

    @abstractmethod
    def load_text_dataset(self) -> TextDataset:
        raise NotImplementedError


class TextDataset(ABC):
    @property
    @abstractmethod
    def split_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def iter_texts(self, split_name: str):
        raise NotImplementedError


class DatasetDictTextDataset(TextDataset):
    def __init__(self, dataset: DatasetDict) -> None:
        self.dataset = dataset

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(self.dataset.keys())

    def iter_texts(self, split_name: str):
        if split_name not in self.dataset:
            raise ValueError(f"Split '{split_name}' not found.")

        for row in self.dataset[split_name]:
            text = row.get("text", "")
            if text is None:
                continue
            yield str(text)


@dataclass(frozen=True)
class ByteRangeSplit:
    start: int
    end: int


class ByteSliceTextDataset(TextDataset):
    def __init__(self, file_path: Path, split_ranges: dict[str, ByteRangeSplit]) -> None:
        self.file_path = file_path
        self._split_ranges = split_ranges

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(self._split_ranges.keys())

    def iter_texts(self, split_name: str):
        if split_name not in self._split_ranges:
            raise ValueError(f"Split '{split_name}' not found.")

        split_range = self._split_ranges[split_name]
        raw_bytes = self.file_path.read_bytes()[split_range.start : split_range.end]
        yield raw_bytes.decode("latin-1")


class HuggingFaceDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name: str, dataset_path: str, dataset_config: str | None = None) -> None:
        super().__init__(dataset_name)
        self.source_path = dataset_path
        self.source_config = dataset_config

    def load(self) -> DatasetDict:
        if self.dataset_path.exists():
            return load_from_disk(str(self.dataset_path))

        if self.source_config is None:
            dataset = load_dataset(self.source_path)
        else:
            dataset = load_dataset(self.source_path, self.source_config)

        save_dataset(self.dataset_name, dataset)
        return dataset

    def load_text_dataset(self) -> TextDataset:
        return DatasetDictTextDataset(self.load())


class Enwik8DatasetLoader(DatasetLoader):
    def load(self) -> Path:
        legacy_file_path = self.dataset_path
        dataset_dir = self.dataset_path.parent / "enwik8"
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

    def load_text_dataset(self) -> TextDataset:
        split_ranges = {
            split_name: ByteRangeSplit(start=start, end=end)
            for split_name, (start, end) in ENWIK8_SPLIT_RANGES.items()
        }
        return ByteSliceTextDataset(self.load(), split_ranges)


class OneBillionWordDatasetLoader(DatasetLoader):
    def load(self) -> DatasetDict:
        if self.dataset_path.exists():
            return load_from_disk(str(self.dataset_path))

        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path = self.dataset_path.parent / ONE_BILLION_WORD_ARCHIVE_NAME
        extract_dir = self.dataset_path.parent / ONE_BILLION_WORD_EXTRACT_DIR_NAME

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

        dataset = load_dataset(
            "text",
            data_files={
                "train": train_files,
                "validation": heldout_files,
                "test": heldout_files,
            },
        )
        save_dataset(self.dataset_name, dataset)
        return dataset

    def load_text_dataset(self) -> TextDataset:
        return DatasetDictTextDataset(self.load())


def build_dataset_loader(dataset_name: str) -> DatasetLoader:
    normalized_name = dataset_name.lower()

    if normalized_name == "enwik8":
        return Enwik8DatasetLoader(normalized_name)

    if normalized_name == "one-billion-word":
        return OneBillionWordDatasetLoader(normalized_name)

    if normalized_name in DATASET_SOURCES:
        dataset_path, dataset_config = DATASET_SOURCES[normalized_name]
        return HuggingFaceDatasetLoader(normalized_name, dataset_path, dataset_config)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_data(dataset_name: str) -> DatasetDict | Path:
    loader = build_dataset_loader(dataset_name)
    return loader.load()


def load_text_dataset(dataset_name: str) -> TextDataset:
    loader = build_dataset_loader(dataset_name)
    return loader.load_text_dataset()


def load(dataset_name: str) -> DatasetDict | Path:
    return load_data(dataset_name)
