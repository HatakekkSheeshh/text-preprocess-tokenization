# Text8 Full N-Gram + BPC Results Analysis

## Scope

This note analyzes the `full` export generated for the `text8` dataset using the count-based n-gram pipeline with Laplace smoothing. The goal of this document is twofold:

1. to verify that the exported results are internally consistent and suitable for reporting
2. to provide a reusable interpretation of the results for the final report

The analysis is based on the files exported under:

- `report_export_text8_full_ngram_bpc/quantitative_comparison.csv`
- `report_export_text8_full_ngram_bpc/prediction_examples.csv`
- `report_export_text8_full_ngram_bpc/sentence_scoring_examples.csv`
- `report_export_text8_full_ngram_bpc/metrics/*.json`

## Sanity Check

The exported `full` results appear internally consistent and are suitable to use as a report reference.

The main checks are as follows:

- All nine runs are present: `word`, `char`, and `BPE` tokenization combined with `1-gram`, `2-gram`, and `3-gram`.
- The runs use the full `text8` splits rather than subset limits.
- The raw character counts are aligned with the dataset definition:
  - training: `90,000,000` characters
  - validation: `5,000,000` characters
  - test: `5,000,000` characters
- Special tokens such as `<eos>` no longer appear in the qualitative prediction output.
- The CSV tables and per-run JSON metrics are mutually consistent.

In other words, there is no obvious evidence of an implementation bug in the final exported results.

## Experimental Context

The `text8` corpus is an aggressively normalized benchmark derived from Wikipedia. It contains only lowercase alphabetic text and spaces, with very limited surface variation. This property strongly influences the behavior of the tokenizers:

- word-level tokenization suffers less from orthographic noise than on raw corpora
- character-level tokenization becomes especially stable because the symbol inventory is extremely small
- BPE still benefits from subword segmentation, but its usual advantage over word-level tokenization may be reduced because the corpus is already very clean

This background is important when interpreting the final numbers.

## Quantitative Summary

The main report table can be summarized as follows.

| Tokenizer | N-gram | Train tokens | Tokenizer fit (s) | Model fit (s) | Val. BPC | Val. PPL | Test BPC | Test PPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Word | 1-gram | 15,301,749 | 3.63 | 8.56 | 1.7472 | 1260.03 | 1.7720 | 1313.83 |
| Word | 2-gram | 15,301,749 | 3.59 | 25.20 | 1.7719 | 1393.74 | 1.8176 | 1580.96 |
| Word | 3-gram | 15,301,749 | 3.78 | 63.48 | 2.2391 | 9402.46 | 2.2940 | 10894.51 |
| Char | 1-gram | 90,000,000 | 5.14 | 45.16 | 4.1213 | 17.40 | 4.1248 | 17.45 |
| Char | 2-gram | 90,000,000 | 5.21 | 92.47 | 3.4327 | 10.80 | 3.4373 | 10.83 |
| Char | 3-gram | 90,000,000 | 5.44 | 140.02 | 2.8646 | 7.28 | 2.8845 | 7.38 |
| BPE | 1-gram | 17,293,940 | 335.31 | 9.60 | 2.0256 | 1474.70 | 2.0706 | 1538.72 |
| BPE | 2-gram | 17,293,940 | 316.20 | 25.59 | 1.8379 | 750.02 | 1.9002 | 840.95 |
| BPE | 3-gram | 17,293,940 | 313.51 | 68.73 | 2.2747 | 3617.94 | 2.3510 | 4155.89 |

## Main Observations

### 1. The full runs are much more stable than the earlier medium-scale experiments

Compared with smaller subset experiments, the full runs produce much more coherent behavior. The validation and test metrics are close to each other across all settings, which suggests that the evaluation is stable enough to support report-level discussion.

This matters because earlier quick or medium runs were primarily useful for debugging the pipeline and checking whether the metrics were meaningful. The full export is the first setting where the results become strong enough to support substantive conclusions.

### 2. Word-level tokenization achieves the best BPC on Text8

The most important result is that the lowest validation and test BPC values are obtained by the word-level models:

- best validation BPC: `1.7472` for `word 1-gram`
- best test BPC: `1.7720` for `word 1-gram`

Even `word 2-gram` remains close to this level, with only a small increase in BPC. This is a notable dataset-specific finding. On a heavily normalized corpus such as `text8`, word-level tokenization is much more competitive than one might expect from raw-text language modeling. Because the benchmark has already removed capitalization, punctuation, and most surface irregularities, the word vocabulary is cleaner and less fragmented than in noisier corpora.

This result is consistent with the earlier EDA observations: `text8` is unusual in that it preserves lexical information while removing much of the orthographic variation that often hurts word-based models.

### 3. Character-level tokenization gives the lowest perplexity, but not the best BPC

Character-level models obtain by far the lowest perplexity values:

- `char 1-gram`: test PPL `17.45`
- `char 2-gram`: test PPL `10.83`
- `char 3-gram`: test PPL `7.38`

However, their BPC values are much higher than those of the word-level and BPE models:

- `char 3-gram` test BPC: `2.8845`
- `word 1-gram` test BPC: `1.7720`
- `bpe 2-gram` test BPC: `1.9002`

This is exactly why BPC is necessary in this project. Perplexity is computed in the tokenizer’s own token space, so character-level perplexity benefits greatly from the extremely small character vocabulary. In contrast, BPC normalizes by the number of original characters and is therefore much more appropriate for comparing tokenization schemes at different granularities.

For the final report, this leads to a clean interpretation:

- character-level tokenization is numerically strong in its own token space
- but BPC indicates that it is not the strongest option when the comparison is normalized by original text length

### 4. BPE reaches its best result at the bigram level

Among the BPE models, the 2-gram configuration is clearly the strongest:

- validation BPC: `1.8379`
- test BPC: `1.9002`
- validation PPL: `750.02`
- test PPL: `840.95`

This is a much better outcome than the BPE 1-gram and 3-gram runs. It also shows that BPE can be competitive on `text8`, but its best behavior appears at a moderate context order rather than at trigram level.

Interestingly, `bpe 2-gram` achieves lower perplexity than all word-level runs, yet its BPC is still worse than the best word-level result. Again, this difference reinforces the idea that BPC and PPL answer slightly different questions:

- PPL reflects uncertainty in the tokenizer’s own token space
- BPC reflects compression or prediction efficiency relative to the original text stream

### 5. Trigram models are hurt by sparsity under Laplace smoothing

A consistent pattern across word-level and BPE tokenization is that the trigram models perform much worse than the corresponding unigram or bigram models:

- `word 3-gram` is substantially worse than `word 1-gram` and `word 2-gram`
- `bpe 3-gram` is substantially worse than `bpe 2-gram`

This is not necessarily a bug. It is a known limitation of count-based n-gram models with add-one smoothing. As the context becomes more specific, sparsity increases sharply. Laplace smoothing then spreads probability mass too broadly across the large vocabulary, especially for word and subword tokenization.

In contrast, character-level trigram still improves over character unigram and bigram because the character vocabulary is extremely small and short local patterns are highly repetitive.

Therefore, the trigram results should be interpreted as evidence of smoothing limitations rather than evidence that longer local context is always harmful.

## Efficiency Analysis

The timing results reveal a strong efficiency contrast between tokenizer families.

### Word-level

- tokenizer fit time is very small: around `3.6` to `3.8` seconds
- model fit time increases with n-gram order, from `8.56` seconds to `63.48` seconds

This makes word-level tokenization a very practical baseline on `text8`.

### Character-level

- tokenizer fit time is also small: around `5.1` to `5.4` seconds
- model fit time is much larger because the token sequence is far longer
  - `45.16` seconds for unigram
  - `140.02` seconds for trigram

So even though character-level tokenization is simple and robust, it pays a substantial computational cost because the sequence length is maximal.

### BPE

- tokenizer fit time is dramatically larger: roughly `313` to `335` seconds
- model fit time is similar to word-level once tokenization is complete

This means BPE’s main efficiency bottleneck in this setup is not n-gram fitting itself, but tokenizer training. For a report discussion, this is an important practical trade-off: BPE may offer better token-space modeling behavior than word-level in some cases, but it is much more expensive to prepare.

## Qualitative Behavior

### Next-token prediction

The prediction examples are broadly sensible.

For the context `the history`:

- word 2-gram predicts `of` most strongly
- word 3-gram also predicts `of` most strongly
- BPE 2-gram and 3-gram behave similarly

This is a reasonable outcome and matches natural English usage in the corpus.

For the context `in the`:

- word and BPE models predict tokens such as `one`, `united`, `world`, and `first`
- these candidates are plausible continuations in encyclopedia-style text

Character-level predictions are also locally plausible, but they are harder to interpret semantically because the model operates over individual characters rather than lexical units.

### Sentence scoring

For word-level and BPE tokenization, the model assigns a much better score to the natural phrase `the history of science` than to the reversed sequence `science of history the`, especially for bigram and trigram:

- word 3-gram:
  - natural phrase PPL: `143.65`
  - reversed phrase PPL: `2908.95`
- BPE 3-gram:
  - natural phrase PPL: `87.59`
  - reversed phrase PPL: `1332.81`

This is a good qualitative sign that the models capture meaningful local word order.

Character-level results are less aligned with this example. In fact, the reversed phrase is scored slightly better than the natural phrase for `char 2-gram` and `char 3-gram`. This should not be interpreted as a failure of the pipeline. Instead, it reflects the fact that character-level models evaluate short local character transitions rather than higher-level lexical or syntactic structure. For that reason, this sentence-pair example should be treated as a tokenizer-dependent illustration rather than a universal proof of model quality.

## Report-Oriented Interpretation

The full `text8` results support the following conclusions.

First, `text8` is a corpus where word-level tokenization remains surprisingly strong. Because the data is already heavily normalized, the main disadvantages of word-based modeling are reduced, and the simplest word-level models can achieve the best BPC.

Second, character-level tokenization is extremely stable and achieves the lowest perplexity, but BPC shows that this does not translate into the best cross-tokenizer compression-style performance. This makes character-level modeling a useful baseline, but not necessarily the preferred tokenizer when fairness across token granularities matters.

Third, BPE becomes competitive at the bigram level and clearly outperforms its unigram and trigram variants. However, its tokenizer fitting cost is much higher than that of word-level or character-level tokenization, which is an important practical drawback.

Fourth, Laplace-smoothed trigram models are too sparse for word-level and BPE tokenization on this dataset. In this project, the strongest configurations on `text8` are therefore not the highest-order models, but rather the simpler unigram or bigram settings.

## Recommended Cautions for the Final Report

- Use **BPC as the main metric** for cross-tokenizer comparison.
- Report **PPL as a complementary metric**, not as the sole basis for ranking tokenizers.
- Do not claim that character-level tokenization is the best tokenizer solely because it has the lowest perplexity.
- Do not present the sentence pair `the history of science` vs. `science of history the` as a universal qualitative success case for every tokenizer; it is most meaningful for word-level and BPE models.
- When discussing BPE, explicitly mention that its tokenizer fit time is much larger than the alternatives.

## Practical Recommendation

If only one tokenizer-model combination were to be highlighted for `text8` under the current count-based n-gram framework, the most defensible choices would be:

- `word 1-gram` as the best BPC result
- `bpe 2-gram` as the strongest BPE configuration
- `char 3-gram` as the strongest character-level configuration in terms of perplexity

This three-way summary gives a balanced picture of the trade-offs and is likely more informative than naming a single universal winner.
