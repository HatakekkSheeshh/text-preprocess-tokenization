from __future__ import annotations

from src.tokenizers.base import BaseTokenizer
from src.tokenizers.bpe_tokenizer import BPETokenizer
from src.tokenizers.char_tokenizer import CharTokenizer
from src.tokenizers.word_tokenizer import WordTokenizer


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
        "Currently implemented tokenizers for language-model experiments are: word, char, bpe."
    )
