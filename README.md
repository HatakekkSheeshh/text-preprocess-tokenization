# Text Preprocess & Tokenization for Language Modeling

## Overview

This project focuses on **text preprocessing** and **tokenization** for **language modeling**. The main goal is to compare different tokenization strategies and analyze how they affect vocabulary size, sequence length, computational efficiency, and model quality.

The repository is currently in the initialization stage. No preprocessing pipeline, model training code, or experimental results have been implemented yet. This README provides a high-level project description based on `docs/job_description.pdf` and `docs/Assignment_NLP252_CC.pdf`.

## Objectives

- Perform exploratory data analysis (EDA) on each dataset.
- Build a text preprocessing and tokenization pipeline.
- Compare three tokenization approaches: **word-level**, **character-level**, and **BPE**.
- Build next-token prediction models for language modeling.
- Evaluate tokenization methods using **Vocabulary Size**, **Sequence Length**, **Computational Efficiency**, and **Perplexity**.

## Planned Datasets

| Dataset | Dataset Name | Characteristics |
| --- | --- | --- |
| One Billion Word | `one-billion-word` | Large-scale data, independent sentences, limited long-range context |
| WikiText-103 | `wikitext-103` | Wikipedia-style text, longer context, rich linguistic content |
| Text8 | `text8` | Heavily normalized text, no punctuation, mostly lowercase |
| Enwik8 | `enwik8` | Raw byte-level data with many special characters |

## How to Run

Load Dataset:

```bash
python main.py --load <dataset_name>
```

Run EDA for Dataset:

```bash
python main.py --eda <dataset_name>
```

Both arguments are optional and default to `None`.

Example:

```bash
python main.py --load wikitext-103
python main.py --eda text8
```

EDA outputs will be saved to `outputs/eda/<dataset_name>/` when the EDA flow is implemented.

Train an LSTM language model:

```bash
python main.py --train-lstm text8 --tokenizer word --sequence-length 128 --epochs 5
```

Evaluate tokenization on a dataset:

```bash
python main.py --eval --dataset text8 --tokenizer word
```

Quick smoke test on a small subset:

```bash
python main.py --train-lstm text8 --tokenizer word --sequence-length 32 --batch-size 8 --epochs 1 --embedding-dim 32 --hidden-dim 64 --max-train-tokens 4096 --max-validation-tokens 1024 --max-test-tokens 1024
```

Training artifacts are saved to:

- `outputs/checkpoints/lstm/`
- `outputs/metrics/lstm/`

Currently implemented:

- `text8`
- `wikitext-103`

## Planned Scope

### 1. Data Selection & EDA

Analyze key properties of each dataset, such as:

- Vocabulary distribution
- Sentence or sequence length
- Punctuation usage
- Linguistic diversity and randomness

### 2. Prediction Model Development

The project is expected to focus on **language modeling** with:

- **N-gram language models** such as bigram and trigram
- **Simple RNN/LSTM models** trained from scratch, without using pre-trained models

Model objectives:

- Predict the next word/token
- Compute **Perplexity (PP)** to evaluate language modeling quality

### 3. Tokenization

Compare three tokenization methods:

#### Word-level Tokenization

- Pros: intuitive and close to natural language units
- Cons: large vocabulary size and **OOV** issues

#### Character-level Tokenization

- Pros: almost no OOV problem and small vocabulary size
- Cons: longer token sequences and potentially slower training

#### BPE (Byte Pair Encoding)

- Pros: balances word-level and character-level tokenization and reduces OOV issues
- Cons: requires tokenizer training before use

### 4. Implementation & Evaluation

Planned evaluation metrics:

- **Vocabulary Size**: number of unique tokens in the vocabulary
- **Sequence Length**: average tokenized sequence length
- **Computational Efficiency**: training time and processing speed
- **Perplexity (PP)**: primary metric for next-token prediction quality

## Current Repository Structure

```text
.
|-- docs/
|   |-- Assignment_NLP252_CC.pdf
|   `-- job_description.pdf
`-- README.md
```

## Current Status

- Dataset loading and EDA pipelines are available for multiple datasets
- Simple word/character tokenizers and an LSTM language-model baseline are implemented
- Full tokenization comparison experiments and final benchmark tables are still in progress
- Project requirement documents are stored in `docs/`

## Next Steps

1. Design the folder structure for source code, data, configs, and experiment reports.
2. Implement EDA and preprocessing pipelines for each dataset.
3. Implement word-level, character-level, and BPE tokenizers.
4. Build an N-gram baseline and a simple RNN/LSTM model.
5. Run experiments, summarize metrics, and write the final comparison report.

## References

- `docs/job_description.pdf`
- `docs/Assignment_NLP252_CC.pdf`
