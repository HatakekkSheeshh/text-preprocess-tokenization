from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Iterable


@dataclass
class TokenizerState:
    tokenizer_type: str
    special_tokens: list[str]
    token_to_id: dict[str, int]
    id_to_token: list[str]
    min_freq: int
    max_vocab_size: int | None


class BaseTokenizer:
    def __init__(
        self,
        *,
        tokenizer_type: str,
        special_tokens: list[str],
        min_freq: int = 1,
        max_vocab_size: int | None = None,
    ) -> None:
        self.tokenizer_type = tokenizer_type
        self.special_tokens = special_tokens
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        self.is_fitted = False

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def boundary_tokens(self) -> list[str]:
        return []

    def iter_tokens_from_texts(self, texts) -> list[str]:
        raise NotImplementedError

    @property
    def pad_token(self) -> str:
        return self.special_tokens[0]

    @property
    def unk_token(self) -> str:
        return self.special_tokens[1]

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def fit_from_texts(self, texts) -> None:
        counter = Counter(self.iter_tokens_from_texts(texts))

        kept_tokens = [
            token
            for token, frequency in counter.most_common()
            if frequency >= self.min_freq and token not in self.special_tokens
        ]

        if self.max_vocab_size is not None:
            available_slots = max(self.max_vocab_size - len(self.special_tokens), 0)
            kept_tokens = kept_tokens[:available_slots]

        self.id_to_token = list(self.special_tokens) + kept_tokens
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        self.is_fitted = True

    def refresh_vocab_from_mapping(self, token_to_id: dict[str, int]) -> None:
        max_token_id = max(token_to_id.values(), default=-1)
        id_to_token = [""] * (max_token_id + 1)
        for token, token_id in token_to_id.items():
            id_to_token[token_id] = token
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.is_fitted = True

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before encoding.")

        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

    def encode_texts(self, texts, *, max_tokens: int | None = None) -> list[int]:
        encoded_tokens: list[int] = []

        for token in self.iter_tokens_from_texts(texts):
            encoded_tokens.append(self.token_to_id.get(token, self.unk_token_id))
            if max_tokens is not None and len(encoded_tokens) >= max_tokens:
                break

        return encoded_tokens

    def decode_ids(self, token_ids: list[int]) -> list[str]:
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before decoding.")

        tokens: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.id_to_token):
                tokens.append(self.unk_token)
            else:
                tokens.append(self.id_to_token[token_id])
        return tokens

    def save(self, path: Path) -> None:
        state = TokenizerState(
            tokenizer_type=self.tokenizer_type,
            special_tokens=self.special_tokens,
            token_to_id=self.token_to_id,
            id_to_token=self.id_to_token,
            min_freq=self.min_freq,
            max_vocab_size=self.max_vocab_size,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


class WordTokenizer(BaseTokenizer):
    def __init__(self, *, min_freq: int = 1, max_vocab_size: int | None = None) -> None:
        super().__init__(
            tokenizer_type="word",
            special_tokens=["<pad>", "<unk>", "<eos>"],
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
        )

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def boundary_tokens(self) -> list[str]:
        return ["<eos>"]

    def iter_tokens_from_texts(self, texts):
        first_non_empty = True

        for text in texts:
            tokens = self.tokenize(text)
            if not tokens:
                continue

            if not first_non_empty:
                for token in self.boundary_tokens():
                    yield token

            for token in tokens:
                yield token

            first_non_empty = False


class CharTokenizer(BaseTokenizer):
    def __init__(self, *, min_freq: int = 1, max_vocab_size: int | None = None) -> None:
        super().__init__(
            tokenizer_type="char",
            special_tokens=["<pad>", "<unk>"],
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
        )

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def boundary_tokens(self) -> list[str]:
        return ["\n"]

    def iter_tokens_from_texts(self, texts):
        first_text = True

        for text in texts:
            tokens = self.tokenize(text)
            if not tokens:
                continue

            if not first_text:
                for token in self.boundary_tokens():
                    yield token

            for token in tokens:
                yield token

            first_text = False


class BPETokenizer(BaseTokenizer):
    def __init__(self, *, min_freq: int = 1, max_vocab_size: int | None = None) -> None:
        super().__init__(
            tokenizer_type="bpe",
            special_tokens=["<pad>", "<unk>", "<eos>"],
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
        )
        self._tokenizer = None

    def _require_tokenizers(self):
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        except ImportError as exc:
            raise ImportError(
                "BPE tokenizer requires the `tokenizers` package. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        return Tokenizer, models, pre_tokenizers, trainers

    def fit_from_texts(self, texts: Iterable[str]) -> None:
        Tokenizer, models, pre_tokenizers, trainers = self._require_tokenizers()

        self._tokenizer = Tokenizer(models.BPE(unk_token=self.unk_token))
        self._tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        trainer = trainers.BpeTrainer(
            vocab_size=self.max_vocab_size or 50_000,
            min_frequency=self.min_freq,
            special_tokens=self.special_tokens,
        )
        self._tokenizer.train_from_iterator(texts, trainer=trainer)

        self.refresh_vocab_from_mapping(self._tokenizer.get_vocab(with_added_tokens=True))

    def encode_texts(self, texts, *, max_tokens: int | None = None) -> list[int]:
        if not self.is_fitted or self._tokenizer is None:
            raise RuntimeError("Tokenizer must be fitted before encoding.")

        encoded_tokens: list[int] = []
        first_non_empty = True
        eos_token_id = self.token_to_id["<eos>"]

        for text in texts:
            if not text:
                continue

            token_ids = self._tokenizer.encode(text).ids
            if not token_ids:
                continue

            if not first_non_empty:
                encoded_tokens.append(eos_token_id)
                if max_tokens is not None and len(encoded_tokens) >= max_tokens:
                    return encoded_tokens[:max_tokens]

            encoded_tokens.extend(token_ids)
            first_non_empty = False

            if max_tokens is not None and len(encoded_tokens) >= max_tokens:
                return encoded_tokens[:max_tokens]

        return encoded_tokens

    def decode_ids(self, token_ids: list[int]) -> list[str]:
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before decoding.")

        return super().decode_ids(token_ids)

    def save(self, path: Path) -> None:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer must be fitted before saving.")

        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))


def build_tokenizer(
    tokenizer_name: str,
    *,
    min_freq: int = 1,
    max_vocab_size: int | None = None,
) -> BaseTokenizer:
    tokenizer_name = tokenizer_name.lower()

    if tokenizer_name == "word":
        return WordTokenizer(min_freq=min_freq, max_vocab_size=max_vocab_size)
    if tokenizer_name == "char":
        return CharTokenizer(min_freq=min_freq, max_vocab_size=max_vocab_size)
    if tokenizer_name == "bpe":
        return BPETokenizer(min_freq=min_freq, max_vocab_size=max_vocab_size)

    raise ValueError(
        f"Unsupported tokenizer: {tokenizer_name}. "
        "Currently implemented tokenizers for LSTM training are: word, char, bpe."
    )
