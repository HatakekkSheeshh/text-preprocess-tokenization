# Text8 EDA for Section 3.2

## 1. Dataset overview

Text8 is an aggressively normalized corpus derived from Wikipedia. In this repository version, each split is stored as a single continuous text stream:

- Train: 90,000,000 characters, 15,301,749 whitespace-delimited words
- Validation: 5,000,000 characters, 848,226 words
- Test: 5,000,000 characters, 855,233 words

This already tells us that `text8` is structurally very different from datasets such as WikiText-103:

- there are no sentence boundaries to rely on
- there is no punctuation to exploit
- there is no uppercase information
- the corpus is almost a pure stream of lowercase words separated by spaces

For section 3.2, this means the EDA should focus on token stream properties rather than sentence-level discourse properties.

## 2. Surface-form characteristics

- Character vocabulary size in every split is only 27, which is consistent with a nearly closed alphabet.
- Punctuation ratio is 0.00%, uppercase ratio is 0.00%, and digit ratio is 0.00% on the training split.
- Space ratio is 17.00%, so the corpus alternates very regularly between alphabetic spans and separators.
- The character set is:  , a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z.
- `double_space_count`, `newline_count`, and `tab_count` are all useful sanity checks for normalization quality. On the training split they are 0, 0, and 0.

Interpretation:

`text8` is already preprocessed extremely heavily before it reaches our pipeline. Because of that, standard cleaning operations such as lowercasing, punctuation removal, or number normalization are either redundant or actively harmful for reproducibility. The right preprocessing strategy here is minimalism.

## 3. Word-level statistics

- Training vocabulary size: 239,974
- Validation vocabulary size: 45,371
- Test vocabulary size: 47,725
- Average word length on train: 4.88 characters
- Median word length on train: 4
- 95th percentile word length on train: 10
- Maximum observed word length on train: 100
- Lexical diversity on train (`vocab_size / num_words`): 0.02
- Rare-word ratio on train (frequency <= 2): 60.64%

The top-word coverage is also informative:

- Top 10 most frequent words cover 24.67% of all training tokens
- Top 100 cover 46.98%
- Top 1,000 cover 67.21%

Interpretation:

The corpus is still large enough to have a long-tail vocabulary, but it is much cleaner and more repetitive than raw Wikipedia text. This usually helps token-based language models converge faster because there are fewer orthographic variants competing for probability mass.

## 4. Notable normalization artifacts

- The most frequent training words start with: the, of, and, one, in, a, to, zero, nine, two.
- Number words such as `one`, `zero`, `nine`, `two`, `eight`, and `five` appear among the most frequent tokens, showing that numeric content survives mostly in verbalized form rather than digit form.
- The standalone token `s` is highly frequent, which strongly suggests that apostrophe-based forms were normalized in a way that detached possessive or contraction remnants.
- The longest observed tokens include: bababadalgharaghtakamminarronnkonnbronntonnerronntuonnthunntrovarrhounawnskawntoohoohoordenenthurnuk, germanhungarianczechpolishruthenianromaniancroatslovakserbslovene, sodiummetadiaminoparadioxyarsenobenzoemethylenesulphoxylate, llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch, hottentottensoldatententententoonstellingsterreinen.

Interpretation:

These examples show that `text8` is not raw natural text. It is a benchmark-oriented normalized stream. That is useful because it reduces surface noise, but it also means some artifacts are baked into the benchmark itself. For a word-level model, those unusually long merged strings inflate the vocabulary tail. For BPE, they are easier to decompose into reusable subword units.

## 5. Split overlap and OOV behavior

Compared with the training vocabulary:

- Validation unique OOV ratio: 14.80%
- Validation token-level OOV ratio: 1.11%
- Test unique OOV ratio: 15.43%
- Test token-level OOV ratio: 1.18%

Interpretation:

Even in a normalized benchmark like `text8`, word-level modeling does not fully escape OOV. The important nuance is whether OOV is concentrated in many rare types or in a meaningful portion of evaluation tokens. If token-level OOV stays low, word-level models can remain competitive despite unseen vocabulary. If it rises, subword tokenization becomes more attractive.

## 6. Implications for preprocessing

Recommended preprocessing for `text8`:

1. Preserve the existing lowercase stream exactly as provided.
2. Tokenize words using whitespace only for the word-level baseline.
3. Keep the space character as a valid token for character-level modeling.
4. Train BPE directly on the already normalized training split; do not add extra normalization unless the whole team agrees to deviate from the benchmark.
5. Build training samples from fixed contiguous windows instead of sentences, because sentence segmentation is not meaningful here.

For sequence construction, the training split alone can produce approximately:

- 119,544 non-overlapping windows of 128 words
- 59,772 non-overlapping windows of 256 words
- 29,886 non-overlapping windows of 512 words

This is a useful design signal for your dataloader and for estimating training time.

## 7. Expected model behavior

### Word-level

Word-level tokenization is likely to be stronger on `text8` than on more raw datasets because the corpus has already removed case, punctuation, and many formatting artifacts. The main weakness is the still-large softmax vocabulary and the remaining OOV words on validation/test.

### Character-level

Character-level modeling will enjoy an extremely small vocabulary and essentially no OOV problem. However, it will need much longer sequences to represent the same amount of information. On `text8`, that usually means slower training and harder long-range dependency learning, especially for small RNN/LSTM models.

### BPE

BPE is likely to be the best compromise. Because `text8` is already normalized, the gain from BPE may be smaller than on noisier corpora, but it should still reduce vocabulary size substantially while avoiding much of the OOV burden of pure word-level modeling.

## 8. Practical hypothesis for the report

If the same model family and capacity are used across tokenizers, a reasonable hypothesis for `text8` is:

- BPE will provide the best trade-off between perplexity and efficiency.
- Word-level may achieve competitive perplexity if the vocabulary cap is generous, but it will be more expensive in the embedding/softmax layers.
- Character-level will be the most robust to unseen forms but probably the slowest and weakest in perplexity unless trained longer or with a stronger architecture.

This hypothesis follows directly from the dataset structure: `text8` is simple at the surface level, but still long-context and vocabulary-heavy enough that token granularity matters.
