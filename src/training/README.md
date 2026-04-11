# src/training/

Experiment runners for the current Task 2 prediction baseline.

Current responsibilities:

- fit tokenizers on the training split
- prepare tokenized train/validation/test streams
- run from-scratch n-gram experiments
- save tokenizer artifacts and experiment metrics

Currently implemented:

- `train_ngram.py` for unigram, bigram, and trigram experiments with Laplace smoothing
