from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import torch
from torch.utils.data import Dataset

from src.datasets.load_data import load_text_dataset
from src.tokenizers import BaseTokenizer

REQUIRED_TEXT_SPLITS = ("train", "validation", "test")


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
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive when provided.")

        self.sequence_length = sequence_length
        self.stride = sequence_length if stride is None else stride
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


def validate_required_splits(dataset_name: str, available_splits: tuple[str, ...]) -> None:
    missing_splits = [split_name for split_name in REQUIRED_TEXT_SPLITS if split_name not in available_splits]
    if missing_splits:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required splits: {', '.join(missing_splits)}. "
            f"Available splits: {', '.join(available_splits)}"
        )


def limit_texts(texts, max_texts: int | None):
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
    text_dataset = load_text_dataset(dataset_name)
    validate_required_splits(dataset_name, text_dataset.split_names)

    split_limits = {
        "train": max_train_tokens,
        "validation": max_validation_tokens,
        "test": max_test_tokens,
    }
    target_splits = tuple(split_name for split_name in REQUIRED_TEXT_SPLITS if split_name in text_dataset.split_names)
    prepared_splits: dict[str, PreparedSplit] = {}

    tokenizer.fit_from_texts(limit_texts(text_dataset.iter_texts("train"), max_fit_texts))

    for split_name in target_splits:
        token_ids = tokenizer.encode_texts(
            text_dataset.iter_texts(split_name),
            max_tokens=split_limits[split_name],
        )
        dataset = LanguageModelingDataset(token_ids, sequence_length=sequence_length, stride=stride)
        prepared_splits[split_name] = PreparedSplit(
            token_ids=token_ids,
            num_tokens=len(token_ids),
            num_sequences=len(dataset),
        )

    return PreparedCorpus(tokenizer=tokenizer, splits=prepared_splits)
