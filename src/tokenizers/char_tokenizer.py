from __future__ import annotations

from src.tokenizers.base import BaseTokenizer


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

    def count_characters_for_token_prefix(self, text: str, token_count: int) -> int:
        if token_count <= 0:
            return 0
        return min(token_count, len(text))

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
