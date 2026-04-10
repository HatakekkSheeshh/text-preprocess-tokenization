# src/tokenizers/

Tokenization code for language-model experiments.

Current structure:
- `base.py`: shared tokenizer state and `BaseTokenizer`
- `word_tokenizer.py`: whitespace-based word tokenizer
- `char_tokenizer.py`: character-level tokenizer
- `bpe_tokenizer.py`: BPE tokenizer using the `tokenizers` library
- `factory.py`: `build_tokenizer(...)`
- `__init__.py`: package exports

All tokenizers expose a consistent interface through `BaseTokenizer`.
