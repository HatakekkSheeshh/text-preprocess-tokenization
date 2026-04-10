from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.datasets.load_data import load
from src.tokenizers import BaseTokenizer


TEXT_SPLITS = ("train", "validation", "test")
ENWIK8_SPLIT_RANGES = {
    "train": (0, 90_000_000),
    "validation": (90_000_000, 95_000_000),
    "test": (95_000_000, 100_000_000),
}


@dataclass
class PreparedSplit:
    token_ids: list[int]
    num_tokens: int
    num_sequences: int


@dataclass
class PreparedCorpus:
    tokenizer: BaseTokenizer
    splits: dict[str, PreparedSplit]


class LanguageModelingDataset(Dataset):
    def __init__(self, token_ids: list[int], sequence_length: int, stride: int | None = None) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.max_start = len(token_ids) - sequence_length - 1

        if self.max_start < 0:
            raise ValueError(
                "Not enough token ids to build one training sample. "
                f"Need at least {sequence_length + 1} tokens, got {len(token_ids)}."
            )

        self.num_sequences = (self.max_start // self.stride) + 1

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        end = start + self.sequence_length

        inputs = self.token_ids[start:end]
        targets = self.token_ids[start + 1 : end + 1]
        return inputs, targets


def read_enwik8_split(file_path: Path, split_name: str) -> str:
    if split_name not in ENWIK8_SPLIT_RANGES:
        raise ValueError(f"Unsupported enwik8 split: {split_name}")

    start, end = ENWIK8_SPLIT_RANGES[split_name]
    raw_bytes = file_path.read_bytes()[start:end]
    return raw_bytes.decode("latin-1")


def iter_split_texts(dataset_name: str, split_name: str):
    dataset = load(dataset_name)

    if dataset_name == "enwik8":
        if split_name not in ENWIK8_SPLIT_RANGES:
            raise ValueError(f"Unsupported split for enwik8: {split_name}")
        yield read_enwik8_split(dataset, split_name)
        return

    if split_name not in dataset:
        raise ValueError(f"Split '{split_name}' not found in dataset '{dataset_name}'.")

    for row in dataset[split_name]:
        text = row.get("text", "")
        if text is None:
            continue
        yield str(text)


def maybe_limit_texts(texts, max_texts: int | None):
    if max_texts is None:
        yield from texts
        return

    yield from islice(texts, max_texts)


def build_prepared_corpus(
    dataset_name: str,
    tokenizer: BaseTokenizer,
    *,
    max_fit_texts: int | None = None,
    max_train_tokens: int | None = None,
    max_validation_tokens: int | None = None,
    max_test_tokens: int | None = None,
    sequence_length: int,
    stride: int | None = None,
) -> PreparedCorpus:
    tokenizer.fit_from_texts(maybe_limit_texts(iter_split_texts(dataset_name, "train"), max_fit_texts))

    split_limits = {
        "train": max_train_tokens,
        "validation": max_validation_tokens,
        "test": max_test_tokens,
    }
    prepared_splits: dict[str, PreparedSplit] = {}

    for split_name in TEXT_SPLITS:
        token_ids = tokenizer.encode_texts(
            iter_split_texts(dataset_name, split_name),
            max_tokens=split_limits[split_name],
        )
        dataset = LanguageModelingDataset(token_ids, sequence_length=sequence_length, stride=stride)
        prepared_splits[split_name] = PreparedSplit(
            token_ids=token_ids,
            num_tokens=len(token_ids),
            num_sequences=len(dataset),
        )

    return PreparedCorpus(tokenizer=tokenizer, splits=prepared_splits)
