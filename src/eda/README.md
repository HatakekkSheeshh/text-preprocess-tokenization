# src/eda/

Code for exploratory data analysis (EDA).

Typical analyses:
- vocabulary statistics
- sequence/sentence length distribution
- punctuation statistics
- dataset-specific text characteristics

Generated plots or tables should be saved to `outputs/eda/`.

## wikitext-103
With containing long-range context and diverse vocabulary, we explore primarily on these aspects:
- Document/sentence length 
- Long-tail vocabulary 
- Punctuation ratio
- Rare words

## text8
Because `text8` is already aggressively normalized into a single lowercase token stream, the EDA focuses on:
- word and character vocabulary statistics
- long-tail vocabulary and rare words
- token overlap between train/validation/test
- normalization artifacts such as detached `s` tokens or unusually long merged words
- implications for word-level, character-level, and BPE tokenization
- report-friendly plots saved to `outputs/eda/text8/plots/` and `docs/figures/text8/`
