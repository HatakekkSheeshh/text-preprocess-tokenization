from __future__ import annotations

from src.tokenizers.base import BaseTokenizer


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
