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

        # init tokenizer
        self._tokenizer = Tokenizer(models.BPE(unk_token=self.unk_token))
        self._tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        # 
        trainer = trainers.BpeTrainer(
            vocab_size=self.max_vocab_size or 50_000,
            min_frequency=self.min_freq,
            special_tokens=self.special_tokens,
        )
        
        list_texts = list(texts)
        self._tokenizer.train_from_iterator(list_texts, trainer=trainer)

        self.refresh_vocab_from_mapping(self._tokenizer.get_vocab(with_added_tokens=True))

    def boundary_tokens(self) -> list[str]:
        return ["<eos>"]

    def encode_text(self, text: str) -> list[int]:
        if not self.is_fitted or self._tokenizer is None:
            raise RuntimeError("Tokenizer must be fitted before encoding.")
        return self._tokenizer.encode(text).ids

    def count_characters_for_token_prefix(self, text: str, token_count: int) -> int:
        if token_count <= 0:
            return 0
        if not self.is_fitted or self._tokenizer is None:
            raise RuntimeError("Tokenizer must be fitted before measuring character coverage.")

        encoding = self._tokenizer.encode(text)
        if token_count >= len(encoding.ids):
            return len(text)
        return encoding.offsets[token_count - 1][1]

    def decode_ids(self, token_ids: list[int]) -> list[str]:
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before decoding.")

        return super().decode_ids(token_ids)

    def save(self, path: Path) -> None:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer must be fitted before saving.")

        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))
