from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.tokenizers.base import BaseTokenizer


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
