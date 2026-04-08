# src/tokenizers/

Code for tokenization methods.

Planned tokenizers:
- word-level tokenizer
- character-level tokenizer
- BPE tokenizer

All tokenizers should ideally expose a consistent interface such as `fit()`, `encode()`, and `decode()`.

Currently implemented for LSTM training:
- whitespace-based word tokenizer
- character-level tokenizer
- BPE tokenizer using the `tokenizers` library
