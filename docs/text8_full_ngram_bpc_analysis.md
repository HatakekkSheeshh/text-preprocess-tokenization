# Text8 Medium and Full N-Gram + BPC Results Analysis

## Scope

This note summarizes and verifies the latest `text8` exports produced by the BPC-aware n-gram notebook. It is intended as a report-writing reference rather than as a raw experiment log.

The analysis covers two exported result bundles:

- `report_export_text8_medium_ngram_bpc/`
- `report_export_text8_full_ngram_bpc/`

Both bundles contain:

- `quantitative_comparison.csv`
- `prediction_examples.csv`
- `sentence_scoring_examples.csv`
- per-run metrics under `metrics/`
- tokenizer/model artifacts under `artifacts/`

## Verification Verdict

The current `medium` and `full` exports look internally consistent and are suitable to use as report references.

The main verification points are:

- All nine runs are present in each bundle: `word`, `char`, and `bpe`, each combined with `1-gram`, `2-gram`, and `3-gram`.
- The `medium` export now uses aligned raw-text budgets across tokenizers:
  - training: `1,000,000` characters
  - validation: `250,000` characters
  - test: `250,000` characters
- The `full` export uses the entire `text8` corpus:
  - training: `90,000,000` characters
  - validation: `5,000,000` characters
  - test: `5,000,000` characters
- Qualitative prediction outputs no longer contain special tokens such as `<eos>`.
- The empty `sentence_scoring_examples.csv` files are expected in these exports, because the notebook currently sets `score_texts = []` and focuses on next-token prediction instead.

No obvious implementation bug is visible from the current exported results.

## Why These Results Matter

`text8` is a special dataset for tokenizer comparison. It is already aggressively normalized: the text is lowercase, punctuation-free, and stored as a continuous stream of words separated by spaces. Because of this:

- word-level tokenization is less harmed by orthographic variation than on raw corpora
- character-level tokenization becomes extremely stable because the symbol inventory is tiny
- BPE still helps reduce vocabulary sparsity, but its advantage over word-level tokenization is less automatic than on noisier datasets

This dataset property is important for interpreting both `BPC` and `PPL`.

## Medium-Scale Results

The `medium` export is best interpreted as a fair budgeted comparison. Every tokenizer sees the same amount of raw text, so the `BPC` values are much more meaningful than in earlier token-budgeted runs.

### Medium quantitative summary

| Tokenizer | N-gram | Train chars | Tok. fit (s) | Model fit (s) | Val. BPC | Val. PPL | Test BPC | Test PPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Word | 1-gram | 1,000,000 | 0.2546 | 0.1606 | 1.8182 | 1832.29 | 1.8492 | 1875.21 |
| Word | 2-gram | 1,000,000 | 0.2441 | 0.4045 | 2.0834 | 5482.32 | 2.1166 | 5577.81 |
| Word | 3-gram | 1,000,000 | 0.2338 | 1.1234 | 2.2889 | 12815.20 | 2.3244 | 13009.56 |
| Char | 1-gram | 1,000,000 | 0.2521 | 0.9429 | 4.1280 | 17.48 | 4.1243 | 17.44 |
| Char | 2-gram | 1,000,000 | 0.2567 | 1.7971 | 3.4432 | 10.88 | 3.4374 | 10.83 |
| Char | 3-gram | 1,000,000 | 0.2651 | 2.7074 | 2.9397 | 7.67 | 2.9520 | 7.74 |
| BPE | 1-gram | 1,000,000 | 30.1970 | 0.1697 | 2.1644 | 1902.58 | 2.2865 | 1966.03 |
| BPE | 2-gram | 1,000,000 | 28.0132 | 0.3840 | 2.4879 | 5882.13 | 2.6300 | 6144.57 |
| BPE | 3-gram | 1,000,000 | 27.2096 | 1.1598 | 2.7077 | 12662.70 | 2.8544 | 12932.04 |

### Medium interpretation

The medium run shows three useful patterns.

First, `word 1-gram` already achieves the best `BPC` in this budgeted setting. This suggests that `text8` is indeed unusually friendly to word-level modeling.

Second, `char 3-gram` achieves the lowest `PPL`, but not the best `BPC`. This confirms that perplexity alone is not a fair cross-tokenizer ranking metric.

Third, BPE is still underwhelming at medium scale. In this export, even `bpe 1-gram` is better than `bpe 2-gram` and `bpe 3-gram`, which suggests that the BPE model needs more data before higher-order contexts become useful.

For the report, the medium results are best used as a controlled comparison that motivated the move to a full-scale run.

## Full-Scale Results

The `full` export is the stronger basis for the final report because it uses the entire `text8` corpus and produces more stable behavior.

### Full quantitative summary

| Tokenizer | N-gram | Train chars | Tok. fit (s) | Model fit (s) | Val. BPC | Val. PPL | Test BPC | Test PPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Word | 1-gram | 90,000,000 | 4.1726 | 12.8609 | 1.7472 | 1260.03 | 1.7720 | 1313.83 |
| Word | 2-gram | 90,000,000 | 4.1765 | 31.1703 | 1.7719 | 1393.74 | 1.8176 | 1580.96 |
| Word | 3-gram | 90,000,000 | 4.0902 | 71.3331 | 2.2391 | 9402.46 | 2.2940 | 10894.51 |
| Char | 1-gram | 90,000,000 | 6.8667 | 69.5887 | 4.1213 | 17.40 | 4.1248 | 17.45 |
| Char | 2-gram | 90,000,000 | 6.8687 | 134.6101 | 3.4327 | 10.80 | 3.4373 | 10.83 |
| Char | 3-gram | 90,000,000 | 6.8882 | 205.0005 | 2.8646 | 7.28 | 2.8845 | 7.38 |
| BPE | 1-gram | 90,000,000 | 190.7295 | 14.6333 | 2.0256 | 1474.70 | 2.0706 | 1538.72 |
| BPE | 2-gram | 90,000,000 | 187.5740 | 32.5649 | 1.8379 | 750.02 | 1.9002 | 840.95 |
| BPE | 3-gram | 90,000,000 | 108.1356 | 88.0856 | 2.2747 | 3617.94 | 2.3510 | 4155.89 |

### Full interpretation

The full run supports several stable conclusions.

### 1. Word-level tokenization remains the strongest in BPC

The best `BPC` values come from the word-level models:

- best validation BPC: `1.7472` for `word 1-gram`
- best test BPC: `1.7720` for `word 1-gram`

Even `word 2-gram` remains close. This is a strong dataset-specific result: on `text8`, once the benchmark has already normalized away most surface noise, a simple word-level model can be highly competitive.

### 2. Character-level tokenization still dominates perplexity

Character-level models have the lowest `PPL` values by a large margin:

- `char 1-gram` test PPL: `17.45`
- `char 2-gram` test PPL: `10.83`
- `char 3-gram` test PPL: `7.38`

However, their `BPC` values remain much worse than those of the strongest word-level and BPE models. This is the clearest evidence that `BPC` should be the primary metric in the final cross-tokenizer comparison.

### 3. BPE improves substantially at full scale, especially for bigram

The most interesting change from `medium` to `full` is BPE behavior. At medium scale, `bpe 1-gram` was the best BPE setting. At full scale, `bpe 2-gram` clearly becomes the strongest BPE configuration:

- validation BPC: `1.8379`
- test BPC: `1.9002`
- validation PPL: `750.02`
- test PPL: `840.95`

This suggests that BPE needs more data before its subword structure becomes useful in a count-based model.

### 4. Trigram remains too sparse for word and BPE under Laplace smoothing

Both `word 3-gram` and `bpe 3-gram` are much worse than their unigram or bigram counterparts. This is consistent with sparsity rather than with a coding bug. Under add-one smoothing, higher-order count-based models can spread probability mass too thinly over a large vocabulary. Character-level trigram still improves because the character vocabulary is tiny and local patterns repeat much more often.

## Medium-to-Full Consistency

The most reassuring aspect of the new exports is that the broad qualitative story is stable across scales:

- `word` remains strongest by `BPC`
- `char` remains strongest by `PPL`
- `trigram` remains harmful for `word` and `bpe`
- `BPC` and `PPL` clearly reward different aspects of the models

At the same time, the full run reveals a more mature BPE pattern than the medium run. This is exactly what we want from the full experiment: it keeps the high-level story, but removes some of the instability caused by small budgets.

## Efficiency Discussion

The timing results show a clear trade-off.

### Word

- tokenizer fitting is very cheap
- model fitting is also relatively cheap
- this makes word-level tokenization the simplest and most practical baseline on `text8`

### Character

- tokenizer fitting is also cheap
- model fitting is slower because the token sequence is much longer
- the cost rises sharply from unigram to trigram

### BPE

- tokenizer fitting is by far the most expensive step
- model fitting itself is not dramatically worse than word-level fitting
- the main overhead comes from learning the tokenizer, not from counting n-grams

This means BPE should be discussed not only in terms of quality, but also in terms of preprocessing cost.

## Qualitative Behavior

The current notebook now uses raw-text next-token prediction as the qualitative example, which is a better fit for comparing different tokenization granularities than the earlier sentence-pair check.

The full prediction examples are sensible for word-level and BPE models:

- after `the history `, both `word` and `bpe` bigram/trigram predict `of`
- after `united `, both predict `states`
- after `world war `, both predict `ii`

These are strong qualitative examples because they reflect common encyclopedia-style continuations in `text8`.

Character-level predictions are still reasonable, but they should be interpreted differently. Since the model predicts a single character rather than a word or subword unit, its top predictions are short local continuations such as a space or a frequent letter. This is expected and should not be treated as a failure of the character-level pipeline.

In other words:

- next-token prediction is a good qualitative illustration for all three tokenizers
- but the semantic interpretability of the top prediction is naturally stronger for `word` and `bpe` than for `char`

## Report-Oriented Conclusions

The current exports support the following report-ready conclusions.

First, `text8` is a corpus on which word-level tokenization remains unusually competitive. Because the dataset is already normalized, the disadvantages of word-based modeling are reduced, and the best `BPC` results come from the simplest word-level configurations.

Second, character-level tokenization achieves the lowest perplexity but not the best `BPC`. This shows why `PPL` should not be used as the only ranking criterion across tokenizers with different granularities.

Third, BPE becomes much more competitive when the full corpus is used. In particular, `bpe 2-gram` is clearly stronger than `bpe 1-gram` and `bpe 3-gram` at full scale, even though this pattern is not yet visible in the medium experiment.

Fourth, Laplace-smoothed trigram models are too sparse for `word` and `bpe` on this dataset. The strongest report candidates are therefore `word 1-gram`, `word 2-gram`, `bpe 2-gram`, and `char 3-gram`, rather than the trigram models across the board.

## Recommended Use in the Final Report

The safest way to use these exports in the report is:

- use `BPC` as the main metric for cross-tokenizer comparison
- use `PPL` as a complementary metric
- use the `full` table as the main evidence
- use the `medium` table as a smaller supporting experiment or sanity-check
- use next-token prediction as the qualitative illustration
- avoid over-claiming from character-level next-token predictions, since they operate in a different token space

## Short Final Verdict

The current `medium` and `full` exports are in good shape and are suitable to support the `text8` analysis in the final report.

If one concise summary is needed, the most defensible choices are:

- `word 1-gram` as the best overall `BPC` result on `text8`
- `bpe 2-gram` as the strongest BPE configuration at full scale
- `char 3-gram` as the strongest character-level configuration in terms of perplexity

This three-part summary is more accurate than naming a single universal winner.
