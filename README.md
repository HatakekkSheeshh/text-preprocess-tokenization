# Text Preprocess & Tokenization for Language Modeling

This project provides a simple pipeline for loading text datasets, running EDA, evaluating tokenizers, and training a basic n-gram language model for language modeling experiments.

## What Is Included

- Download and cache datasets under `data/raw/`
- Read processed datasets from `data/processed/` for tokenizer evaluation and n-gram training
- Run EDA for supported datasets and save results under `outputs/eda/`
- Evaluate `word`, `char`, and `bpe` tokenizers
- Train `unigram`, `bigram`, and `trigram` language models
- Support Laplace smoothing, next-token prediction, sentence scoring, Bits Per Character (BPC), and perplexity computation

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
- `all`

Use `all` to run a command across every supported dataset.

## Smoke Mode

Use `--smoke` to apply small dataset-specific limits for a quick test run.

Smoke mode only fills in limits that you did not set manually. For example, if you pass `--max-train-tokens 50000`, that value is kept.

| Dataset | `--max-fit-texts` | `--max-fit-characters` | `--max-eval-texts-per-split` | `--max-train-tokens` | `--max-validation-tokens` | `--max-test-tokens` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `text8` | `1000` | `200000` | `1000` | `4096` | `1024` | `1024` |
| `wikitext-103` | `1000` | - | `500` | `10000` | `2000` | `2000` |
| `enwik8` | `1` | `200000` | `1` | `20000` | `5000` | `5000` |
| `one-billion-word` | `1000` | - | `500` | `20000` | `5000` | `5000` |

Quick smoke test on one dataset:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --smoke
```

Quick smoke test on all datasets:

```bash
python main.py --train-ngram all --tokenizer word --ngram-order 3 --smoke
```

## Commands in main.py

### 1. Load a Dataset

Download and save a dataset to `data/raw/`:

```bash
python main.py --load text8
```

Other examples:

```bash
python main.py --load wikitext-103
python main.py --load enwik8
python main.py --load one-billion-word
```

Load all supported datasets:

```bash
python main.py --load all
```

### 2. Run EDA

Run EDA for a downloaded dataset:

```bash
python main.py --eda text8
```

Supported EDA datasets:

```bash
python main.py --eda text8
python main.py --eda wikitext-103
python main.py --eda enwik8
python main.py --eda one-billion-word
```

Run EDA for all supported datasets:

```bash
python main.py --eda all
```

EDA outputs are saved to:

```text
outputs/eda/<dataset_name>/
```

### 3. Evaluate a Tokenizer

Evaluate a tokenizer on a dataset. Evaluation reads datasets from `data/processed/`:

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

Example:

```bash
python main.py --eval --dataset text8 --tokenizer word --max-vocab-size 10000 --smoke
python main.py --eval --dataset text8 --tokenizer word --max-vocab-size 30000 --smoke
python main.py --eval --dataset text8 --tokenizer word --max-vocab-size 50000 --smoke

python main.py --eval --dataset text8 --tokenizer bpe --max-vocab-size 4000 --smoke
python main.py --eval --dataset text8 --tokenizer bpe --max-vocab-size 8000 --smoke
python main.py --eval --dataset text8 --tokenizer bpe --max-vocab-size 16000 --smoke

```

Evaluate one tokenizer on all datasets with smoke limits:

```bash
python main.py --eval --dataset all --tokenizer word --smoke
```

Tokenizer evaluation results are saved to:

```text
outputs/metrics/tokenization/
```

### 4. Train an N-Gram Language Model

Train a trigram model with the word tokenizer. Training reads datasets from `data/processed/`:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --laplace-alpha 1.0
```

Run a quick smoke test on a small subset:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --max-train-tokens 4096 --max-validation-tokens 1024 --max-test-tokens 1024
```

For single-stream corpora such as `text8` and `enwik8`, limiting the number of fit texts may still leave you fitting on one very large text. In those cases, `--max-fit-characters` is the safer limit:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --max-fit-characters 200000 --max-train-tokens 4096 --max-validation-tokens 1024 --max-test-tokens 1024
```

The same smoke test can now be written with:

```bash
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --smoke
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

Train the same n-gram configuration on all supported datasets:

```bash
python main.py --train-ngram all --tokenizer word --ngram-order 3 --smoke
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

Use `--score-text` to compute log probability, BPC, and perplexity for a text string:

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

On `text8`, you can also cap the fitted corpus by characters:

```bash
python main.py --train-ngram text8 --tokenizer bpe --ngram-order 3 --max-fit-characters 200000 --max-train-tokens 4096 --max-validation-tokens 1024 --max-test-tokens 1024
```

## All main.py Options

| Option | Description | Default |
| --- | --- | --- |
| `--load` | Download and cache a dataset, or use `all` | `None` |
| `--eda` | Run EDA for a dataset, or use `all` | `None` |
| `--eval` | Enable tokenizer evaluation mode | `False` |
| `--dataset` | Dataset name used with `--eval`, or `all` | `None` |
| `--train-ngram` | Train an n-gram language model on a dataset, or use `all` | `None` |
| `--tokenizer` | Tokenizer type: `word`, `char`, or `bpe` | `word` |
| `--ngram-order` | N-gram order: `1`, `2`, or `3` | `3` |
| `--laplace-alpha` | Laplace smoothing coefficient | `1.0` |
| `--min-freq` | Minimum token frequency kept in the vocabulary | `1` |
| `--max-vocab-size` | Maximum vocabulary size | `50000` |
| `--max-fit-texts` | Maximum number of texts used to fit the tokenizer | `None` |
| `--max-fit-characters` | Maximum number of raw characters used to fit the tokenizer | `None` |
| `--max-eval-texts-per-split` | Maximum number of texts per split for tokenizer evaluation | `None` |
| `--max-train-tokens` | Maximum number of training tokens for n-gram training | `None` |
| `--max-validation-tokens` | Maximum number of validation tokens | `None` |
| `--max-test-tokens` | Maximum number of test tokens | `None` |
| `--run-name` | Custom run name | `None` |
| `--smoke` | Use small dataset-specific limits for a quick smoke test | `False` |
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
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --smoke
```

After that, run fuller experiments:

```bash
python main.py --eval --dataset text8 --tokenizer word
python main.py --train-ngram text8 --tokenizer word --ngram-order 3 --predict-context "the history of" --score-text "the history of science"
```
