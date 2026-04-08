# Text8 LSTM Results Analysis

## Scope

This note summarizes the current LSTM language modeling results on the `text8` dataset. It is intended as a draft analysis that can later be adapted into the final report. The experiments cover:

- one larger `word`-level run on the full training split
- one controlled medium-scale comparison across `word`, `char`, and `BPE` tokenization

All numbers below are taken from the exported metric files in `outputs/metrics/`.

## Experimental Setup

### 1. Full word-level run

Run name: `colab_text8_lstm_word_full`

- Dataset: `text8`
- Tokenizer: `word`
- Vocabulary size: `50,000`
- Sequence length: `128`
- Batch size: `64`
- Embedding dimension: `256`
- Hidden dimension: `512`
- Number of layers: `2`
- Dropout: `0.2`
- Epochs: `5`
- Device: `cuda`
- Number of parameters: `42,128,208`
- Training tokens: `15,301,749`

### 2. Medium comparison across tokenizers

Run names:

- `colab_compare_word_medium`
- `colab_compare_char_medium`
- `colab_compare_bpe_medium`

Shared configuration:

- Dataset: `text8`
- Sequence length: `128`
- Batch size: `32`
- Embedding dimension: `128`
- Hidden dimension: `256`
- Number of layers: `2`
- Dropout: `0.2`
- Epochs: `2`
- Device: `cuda`
- Training tokens: `300,000`
- Validation tokens: `50,000`
- Test tokens: `50,000`

Tokenizer-specific settings:

- `word`: vocabulary capped at `20,000`
- `BPE`: vocabulary capped at `20,000`
- `char`: no vocabulary cap, resulting vocabulary size `29`

## Quantitative Results

### Full word-level run

| Run | Tokenizer | Vocab size | Parameters | Best val loss | Best val perplexity | Test loss | Test perplexity | Training time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `colab_text8_lstm_word_full` | word | 50,000 | 42,128,208 | 4.9638 | 143.14 | 5.1908 | 179.61 | 937.28 s |

Validation performance improved steadily over training:

| Epoch | Train perplexity | Validation perplexity |
| --- | ---: | ---: |
| 1 | 853.52 | 361.95 |
| 2 | 360.09 | 222.75 |
| 3 | 253.74 | 177.26 |
| 4 | 207.19 | 154.97 |
| 5 | 180.21 | 143.14 |

### Medium tokenizer comparison

| Run | Tokenizer | Vocab size | Parameters | Val perplexity | Test perplexity | Training time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `colab_compare_word_medium` | word | 20,000 | 8,621,600 | 1211.77 | 1107.49 | 4.00 s |
| `colab_compare_char_medium` | char | 29 | 932,765 | 7.53 | 7.37 | 2.23 s |
| `colab_compare_bpe_medium` | bpe | 20,000 | 8,621,600 | 2109.67 | 2197.27 | 4.87 s |

## Interpretation

### 1. The full word-level run shows that the LSTM baseline is learning meaningful structure on Text8

The full `word`-level experiment shows a clear and consistent reduction in validation perplexity from `361.95` to `143.14` across five epochs. This indicates that the LSTM baseline is not merely functioning technically, but is actually learning useful sequential patterns from the corpus. The corresponding test perplexity of `179.61` is much stronger than the earlier smoke-test results, confirming that the pipeline behaves reasonably once the model is trained with a realistic amount of data.

This result is also consistent with the earlier EDA findings for `text8`. Because the corpus is aggressively normalized, lowercased, and punctuation-free, the word-level baseline is not penalized as heavily by orthographic noise as it would be on a raw corpus. In other words, `text8` is unusually friendly to a word-based model compared with noisier datasets such as WikiText-103 or One Billion Word.

### 2. The medium comparison is useful, but it must be interpreted carefully

The medium comparison was designed as a controlled sanity check rather than a definitive ranking of tokenizers. All three runs used the same nominal sequence length, hidden size, and number of epochs, but the token spaces are fundamentally different. A sequence length of `128` means:

- `128` words for the word-level run
- `128` characters for the character-level run
- `128` subword units for the BPE run

Therefore, each model sees a different amount of underlying text per training example. The output vocabularies also differ dramatically, which changes the difficulty of the softmax layer and the total number of model parameters. For this reason, the perplexity values should not be treated as perfectly comparable in the same way one would compare two models trained on the same tokenization.

### 3. Character-level tokenization achieves the lowest perplexity in the medium comparison, but this is not enough to conclude that it is the best tokenizer

The `char` model obtains the lowest validation and test perplexity (`7.53` and `7.37`) while also training the fastest (`2.23` seconds). However, this result must be interpreted in the context of the much smaller vocabulary (`29`) and much smaller model size (`932,765` parameters). Character-level perplexity is computed over a very small token inventory, so its numerical scale is not directly comparable to word-level or BPE perplexity.

Even so, the result does support one important conclusion: character-level modeling is extremely robust on `text8`. This is expected because the dataset is already normalized to lowercase alphabetic text plus spaces, leaving almost no tokenization ambiguity and almost no out-of-vocabulary risk. In this setting, character modeling becomes a very strong baseline.

### 4. Under the current budget, word-level tokenization performs better than BPE

In the medium comparison, the `word` run outperforms the `BPE` run by a large margin:

- word validation perplexity: `1211.77`
- BPE validation perplexity: `2109.67`

This suggests that, under the current training budget, BPE is underperforming rather than showing its full potential. There are several plausible reasons for this:

- the training budget is very small for learning good subword dynamics (`300,000` training tokens and only `2` epochs)
- the BPE vocabulary size (`20,000`) may not yet be well matched to this setup
- `text8` is already so normalized that the usual advantages of subword segmentation may be less immediate than in noisier corpora

In other words, the current results do not show that BPE is inherently unsuitable for `text8`; instead, they suggest that BPE may need more tuning or more training to become competitive with the word-level baseline.

### 5. The current evidence supports a nuanced conclusion rather than a single winner

Taken together, the experiments suggest the following:

- `word` tokenization is a strong and competitive choice on `text8`, especially when trained at a realistic scale
- `char` tokenization is extremely stable and efficient on this dataset, but its perplexity should not be directly compared with word/subword perplexity without qualification
- `BPE` has not yet reached competitive performance in the current medium-scale experiment, but that may reflect limited training budget rather than a fundamental weakness

This is an important dataset-specific observation. On a heavily normalized benchmark such as `text8`, word-level tokenization is less disadvantaged than it would be on raw text, while character-level tokenization becomes unusually clean and practical. As a result, the expected advantage of BPE may be smaller or may appear only after additional tuning.

## Report-Oriented Takeaways

The current results support three main statements that can be used in the final report.

First, the full word-level run demonstrates that a standard LSTM language model can learn `text8` effectively when trained on the full corpus, reaching a best validation perplexity of `143.14` and a test perplexity of `179.61`. This makes the LSTM baseline credible enough for subsequent tokenizer comparisons.

Second, the medium-scale comparison confirms that tokenizer choice strongly affects both optimization behavior and efficiency. Character-level modeling is the fastest and numerically achieves the lowest perplexity, but this result is heavily influenced by the tiny character vocabulary and should not be interpreted as a strictly fair win over word or BPE tokenization.

Third, the experiments suggest that `text8` is a special case among language modeling corpora: because it is already aggressively normalized, word-level tokenization remains surprisingly competitive, while BPE may require more training budget or hyperparameter tuning before its usual benefits become visible.

## Recommended Cautions for the Final Report

- Do not claim that `char` is the best tokenizer solely because it has the lowest perplexity.
- Do not compare raw perplexity values across different tokenization granularities without an explicit caveat.
- Make it clear that the `word` full run and the `word/char/BPE` medium runs are not directly comparable because they use different data budgets and model sizes.
- Present the medium comparison as an informative controlled experiment, not as the final definitive ranking of tokenizers.

## Source Files

The analysis above is based on the following local outputs:

- `outputs/metrics/colab_text8_lstm_word_full.json`
- `outputs/metrics/colab_compare_word_medium.json`
- `outputs/metrics/colab_compare_char_medium.json`
- `outputs/metrics/colab_compare_bpe_medium.json`
