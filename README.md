# Text Preprocess & Tokenization for Language Modeling

## Current Focus

The current implementation is scoped to **Task 2: build a basic prediction model**.

Instead of a neural model, the repository now provides a simple **from-scratch n-gram language model** with:

- `unigram`
- `bigram`
- `trigram`
- Laplace smoothing
- next-token prediction
- sentence scoring
- perplexity computation

This keeps the model simple and aligned with the assignment direction for the prediction task.

## Main Command

Train an n-gram language model:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --laplace-alpha 1.0
```

Quick smoke test on a small subset:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --max-train-tokens 4096 --max-validation-tokens 1024 --max-test-tokens 1024
```

Next-word prediction from a context:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --predict-context "the history of"
```

Score a sentence and compute perplexity:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --score-text "the history of science"
```

If you test `bpe`, it is a good idea to limit tokenizer fitting time:

```bash
python main.py --train-ngram text8 --tokenizer bpe --ngram-order 3 --max-fit-texts 1000 --max-train-tokens 4096 --max-validation-tokens 1024 --max-test-tokens 1024
```

## Outputs

N-gram experiment outputs are saved to:

- `outputs/metrics/ngram/`
- `outputs/artifacts/ngram/`

## Implemented Files for Task 2

- `src/models/ngram.py`
- `src/training/train_ngram.py`
- `main.py`

## Notes

- The repository still contains dataset loading, EDA, and tokenizer modules because they are part of the broader project.
- However, the **current modeling scope** is only the basic prediction model for Task 2.
- Tokenization comparison and broader evaluation can be built on top of this baseline later.
