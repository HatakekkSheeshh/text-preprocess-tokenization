from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import json


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
