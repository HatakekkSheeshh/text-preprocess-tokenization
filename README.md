# Text Preprocess & Tokenization for Language Modeling

This project provides a simple pipeline for loading text datasets, running EDA, evaluating tokenizers, and training a basic n-gram language model for language modeling experiments.

## What Is Included

- Download and cache datasets under `data/raw/`
- Run EDA for supported datasets and save results under `outputs/eda/`
- Evaluate `word`, `char`, and `bpe` tokenizers
- Train `unigram`, `bigram`, and `trigram` language models
- Support Laplace smoothing, next-token prediction, sentence scoring, and perplexity computation

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Supported Datasets

Use these dataset names in commands:

- `text8`
- `wikitext-103`
- `enwik8`
- `one-billion-word`

## Commands in main.py

### 1. Load a Dataset

Download and save a dataset to `data/raw/`:

```bash
python main.py --load <dataset_name>
```

Other examples:

```bash
python main.py --load text8
```

### 2. Run EDA

Run EDA for a downloaded dataset:

```bash
python main.py --eda <dataset_name>
```

Supported EDA datasets:

```bash
python main.py --eda text8
```

EDA outputs are saved to:

```text
outputs/eda/<dataset_name>/
```

### 3. Evaluate a Tokenizer

Evaluate a tokenizer on a dataset:

```bash
python main.py --eval --dataset text8 --tokenizer word
```

Supported tokenizers:

- `word`
- `char`
- `bpe`

Example with the character tokenizer:

```bash
python main.py --eval --dataset text8 --tokenizer char
```

Example with the BPE tokenizer:

```bash
python main.py --eval --dataset text8 --tokenizer bpe --max-fit-texts 1000 --max-eval-texts-per-split 1000
```

Tokenizer evaluation results are saved to:

```text
outputs/metrics/tokenization/
```

### 4. Train an N-Gram Language Model

Train a trigram model with the word tokenizer:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --laplace-alpha 1.0
```

Run a quick smoke test on a small subset:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --max-train-tokens 4096 --max-validation-tokens 1024 --max-test-tokens 1024
```

Train a unigram model:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 1
```

Train a bigram model:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 2
```

Train a trigram model:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3
```

### 5. Predict the Next Token

Use `--predict-context` to predict the next token from a context:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --predict-context "the history of"
```

You can pass multiple contexts in one run:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --predict-context "the history of" --predict-context "in the beginning"
```

Change the number of returned predictions with `--top-k`:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --predict-context "the history of" --top-k 10
```

### 6. Score Text and Compute Perplexity

Use `--score-text` to compute log probability and perplexity for a text string:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --score-text "the history of science"
```

You can score multiple strings in one run:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --score-text "the history of science" --score-text "this is a test"
```

### 7. Train with BPE

BPE can take longer to fit, so it is useful to limit the number of fit texts and training tokens during testing:

```bash
python main.py --train-ngram text8 --tokenizer bpe --ngram-order 3 --max-fit-texts 1000 --max-train-tokens 4096 --max-validation-tokens 1024 --max-test-tokens 1024
```

## All main.py Options

| Option | Description | Default |
| --- | --- | --- |
| `--load` | Download and cache a dataset | `None` |
| `--eda` | Run EDA for a dataset | `None` |
| `--eval` | Enable tokenizer evaluation mode | `False` |
| `--dataset` | Dataset name used with `--eval` | `None` |
| `--train-ngram` | Train an n-gram language model on a dataset | `None` |
| `--tokenizer` | Tokenizer type: `word`, `char`, or `bpe` | `word` |
| `--ngram-order` | N-gram order: `1`, `2`, or `3` | `3` |
| `--laplace-alpha` | Laplace smoothing coefficient | `1.0` |
| `--min-freq` | Minimum token frequency kept in the vocabulary | `1` |
| `--max-vocab-size` | Maximum vocabulary size | `50000` |
| `--max-fit-texts` | Maximum number of texts used to fit the tokenizer | `None` |
| `--max-eval-texts-per-split` | Maximum number of texts per split for tokenizer evaluation | `None` |
| `--max-train-tokens` | Maximum number of training tokens for n-gram training | `None` |
| `--max-validation-tokens` | Maximum number of validation tokens | `None` |
| `--max-test-tokens` | Maximum number of test tokens | `None` |
| `--run-name` | Custom run name | `None` |
| `--predict-context` | Context string for next-token prediction; can be used multiple times | `[]` |
| `--score-text` | Text string to score; can be used multiple times | `[]` |
| `--top-k` | Number of predicted tokens to return | `5` |

## Outputs

N-gram training outputs:

```text
outputs/metrics/ngram/
outputs/artifacts/ngram/
```

Tokenizer evaluation outputs:

```text
outputs/metrics/tokenization/
```

EDA outputs:

```text
outputs/eda/
```

## Main Files

- `main.py`: command-line entrypoint
- `src/datasets/load_data.py`: dataset loading and dataset adapters
- `src/eda/`: EDA scripts for supported datasets
- `src/tokenizers/`: word, character, and BPE tokenizers
- `src/evaluation/tokenization.py`: tokenizer evaluation pipeline
- `src/models/ngram.py`: n-gram language model implementation
- `src/training/train_ngram.py`: n-gram training pipeline

## Suggested Workflow

First, download a dataset:

```bash
python main.py --load text8
```

Then run a quick smoke test:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --max-train-tokens 4096 --max-validation-tokens 1024 --max-test-tokens 1024
```

After that, run fuller experiments:

```bash
python main.py --eval --dataset text8 --tokenizer word
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --predict-context "the history of" --score-text "the history of science"
```
