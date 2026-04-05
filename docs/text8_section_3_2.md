# Section 3.2 Draft: Exploratory Data Analysis on Text8

Generated from the local EDA run on April 5, 2026 using [`src/eda/text8.py`](../src/eda/text8.py).

## 3.2.1 Dataset overview

Text8 is a heavily normalized benchmark corpus derived from Wikipedia. Unlike raw encyclopedic text, it is distributed as a continuous lowercase token stream with whitespace-separated words and almost no surface formatting left. In our local copy, each split is stored as a single row of text:

| Split | Characters | Words | Vocabulary Size | Character Vocabulary |
| --- | ---: | ---: | ---: | ---: |
| Train | 90,000,000 | 15,301,749 | 239,974 | 27 |
| Validation | 5,000,000 | 848,226 | 45,371 | 27 |
| Test | 5,000,000 | 855,233 | 47,725 | 27 |

This format already reveals several important properties:

- `text8` is not sentence-oriented; it is a long contiguous stream.
- The corpus contains only spaces and lowercase letters `a-z`.
- There are no punctuation marks, digits, uppercase letters, tabs, or newlines.
- Standard normalization steps such as lowercasing or punctuation stripping have effectively been applied before the data reaches our pipeline.

Because of this structure, EDA for `text8` should emphasize token-stream statistics rather than sentence-level analysis.

## 3.2.2 Surface-form characteristics

The training split confirms that `text8` is aggressively cleaned:

- Space ratio: 17.00%
- Punctuation ratio: 0.00%
- Digit ratio: 0.00%
- Uppercase ratio: 0.00%
- Double-space count: 0
- Newline count: 0
- Tab count: 0

The character set is exactly:

`{space, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z}`

This has two implications. First, the dataset is exceptionally clean and reproducible, which is attractive for controlled tokenization experiments. Second, many preprocessing techniques that would be helpful on noisier corpora are unnecessary here and may even distort the benchmark if applied again.

## 3.2.3 Word-level statistics and lexical structure

On the training split, the average word length is 4.88 characters, the median is 4, and the 95th percentile is 10. However, the vocabulary still has a substantial long tail:

- Rare-word ratio (`frequency <= 2`): 60.64%
- Top 10 most frequent words cover 24.67% of all training tokens
- Top 100 cover 46.98%
- Top 1,000 cover 67.21%
- Lexical diversity (`vocab_size / num_words`): 0.0157

The most frequent words are:

`the, of, and, one, in, a, to, zero, nine, two, is, as, eight, for, s, five, three, was, by, that`

Several observations are useful for later discussion:

- Number words such as `one`, `zero`, `nine`, `two`, `eight`, and `five` are highly frequent, showing that numeric information is preserved mostly in verbalized form instead of digit form.
- The token `s` appears very frequently, which likely comes from apostrophe-based forms after normalization. This is a strong indicator that the corpus is not simply lowercase raw text; it is a transformed benchmark stream.
- Even though the corpus is normalized, the vocabulary remains large enough that word-level softmax layers will still be expensive.

## 3.2.4 Notable artifacts and outliers

The longest observed training tokens are highly unusual:

`bababadalgharaghtakamminarronnkonnbronntonnerronntuonnthunntrovarrhounawnskawntoohoohoordenenthurnuk`  
`germanhungarianczechpolishruthenianromaniancroatslovakserbslovene`  
`sodiummetadiaminoparadioxyarsenobenzoemethylenesulphoxylate`  
`llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch`  
`hottentottensoldatententententoonstellingsterreinen`

The maximum token length in the training split is 100 characters. These examples suggest that `text8` still contains merged titles, transliterated names, and very long lexical items that survive normalization. Therefore, although the dataset is much cleaner than raw Wikipedia, it is not equivalent to ordinary natural prose.

This is an important finding for tokenization analysis:

- Word-level tokenization must keep many rare and very long tokens as atomic units.
- Character-level tokenization handles these outliers naturally, but at the cost of much longer sequences.
- BPE can decompose such outliers into reusable units and is therefore likely to benefit more from these extreme cases than word-level tokenization.

## 3.2.5 Split overlap and OOV analysis

We compared the validation and test vocabularies against the training vocabulary:

| Metric | Validation | Test |
| --- | ---: | ---: |
| Unique OOV words | 6,717 | 7,364 |
| Unique OOV ratio | 14.80% | 15.43% |
| Token-level OOV count | 9,450 | 10,074 |
| Token-level OOV ratio | 1.11% | 1.18% |
| In-vocabulary token ratio | 98.89% | 98.82% |

This is a subtle but valuable result. A relatively large fraction of evaluation word types are unseen in training, but those unseen types account for only a small fraction of total tokens. In other words, most OOV words are rare. This means:

- Word-level models will still suffer from OOV, but the practical damage may be limited if unknown words are infrequent.
- BPE has a clear advantage in type coverage because it can break unseen words into subword units.
- Character-level models effectively avoid OOV altogether, although they pay a price in sequence length and training speed.

## 3.2.6 Implications for preprocessing

For `text8`, the correct preprocessing strategy is intentionally minimal:

1. Preserve the original stream exactly as provided.
2. Use whitespace tokenization for the word-level baseline.
3. Keep the space character as a valid symbol for character-level modeling.
4. Train BPE directly on the normalized training split instead of adding extra normalization rules.
5. Construct training samples from contiguous fixed-length windows rather than sentence boundaries.

From the training split alone, we can derive approximately:

- 119,544 non-overlapping windows of 128 words
- 59,772 non-overlapping windows of 256 words
- 29,886 non-overlapping windows of 512 words

These counts are useful for planning batch construction, context length, and training time.

## 3.2.7 Expected impact on model performance

Based on the dataset characteristics above, we expect the three tokenization strategies to behave differently on `text8`.

Word-level tokenization is likely to perform better here than on raw corpora because the text has already been normalized to a small surface form space: no punctuation, no casing, no digits, and very regular spacing. This should reduce orthographic sparsity and help the model concentrate probability mass. However, the word vocabulary is still large (239,974 on train), and there remains non-zero OOV on validation and test. Therefore, word-level models may achieve competitive perplexity but will likely pay a higher cost in embedding and output layers.

Character-level tokenization should benefit from an extremely small vocabulary of only 27 symbols and essentially no OOV problem. Nevertheless, the sequence length becomes much longer because every word must be represented character by character, including spaces. This will make long-range dependency modeling harder, especially for small RNN/LSTM baselines, and training is likely to be slower.

BPE is the most promising compromise. Because `text8` is already normalized, its gain over word-level tokenization may be smaller than on noisier datasets such as raw Wikipedia. Even so, BPE should still reduce vocabulary size substantially, handle rare or unseen words better than word-level tokenization, and avoid the extreme sequence-length inflation of character-level modeling.

## 3.2.8 Working hypothesis

For this dataset, a reasonable hypothesis is:

- BPE will provide the best overall trade-off between perplexity and computational efficiency.
- Word-level tokenization may remain strong in perplexity because the dataset is already simplified, but it will be less efficient due to the large vocabulary and residual OOV.
- Character-level tokenization will be the most robust to unseen forms but likely the slowest and weakest in perplexity unless trained with more capacity or for longer.

In short, `text8` is an excellent dataset for showing that even a heavily normalized corpus still creates meaningful differences between word, character, and subword tokenization. Its cleanliness reduces noise, but its long stream structure, vocabulary tail, and rare-token behavior continue to make preprocessing and tokenization design matter.
