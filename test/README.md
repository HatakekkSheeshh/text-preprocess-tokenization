# test/

Utility scripts for quick manual checks and small local experiments.

## Current files

- `text_lens.py`: export a few sample `.txt` files from `load_text_dataset(...)` so you can inspect what each split returns.
- `tokenizer_compare.py`: compare `char`, `word`, and `bpe` tokenizers on a dataset split or an external text file.
- `outputs/`: generated files from test scripts.

## `text_lens.py`

This script is useful when you want to verify:

- a dataset can be loaded through `load_text_dataset(...)`
- which splits are exposed by the dataset
- what raw text looks like in `train`, `validation`, and `test`

Supported datasets:

- `one-billion-word`
- `wikitext-103`
- `text8`
- `enwik8`

If `--dataset` is not one of the names above, the script raises a `ValueError`.

### Run

```bash
python test\text_lens.py --dataset enwik8
```

Example with custom limits:

```bash
python test\text_lens.py --dataset text8 --max-texts-per-split 2 --max-chars-per-text 1000
```

### Arguments

- `--dataset`: dataset name to inspect
- `--output-dir`: directory where sample files are written
- `--max-texts-per-split`: number of text files to export for each split
- `--max-chars-per-text`: optional character limit for each exported file

`--max-chars-per-text` is only for quick inspection. It does not affect the real dataset loader or training pipeline.

### Output

Generated files are written to:

```text
test/outputs/load_text_samples/<dataset_name>/<split_name>/text_001.txt
```

Example:

```text
test/outputs/load_text_samples/text8/train/text_001.txt
test/outputs/load_text_samples/text8/validation/text_001.txt
test/outputs/load_text_samples/text8/test/text_001.txt
```

## `tokenizer_compare.py`

This script is useful when you want to quickly compare:

- vocabulary size
- total encoded tokens
- average tokens per text
- a short preview of produced tokens

for the three implemented tokenization strategies:

- `char`
- `word`
- `bpe`

### Run with a dataset

```bash
python test\tokenizer_compare.py --dataset text8 --split train --max-texts 10
```

### Run with an external text file

```bash
python test\tokenizer_compare.py --text-file data\sample.txt --max-texts 20
```

### Arguments

- `--dataset`: dataset name to inspect
- `--text-file`: path to an external `.txt` file
- `--split`: split name when using `--dataset`
- `--max-texts`: maximum number of texts to compare
- `--max-vocab-size`: vocabulary cap passed into the tokenizer
- `--output-dir`: directory where JSON results are written

Output is written to:

```text
test/outputs/tokenizer_compare/<name>.json
```

If the `tokenizers` package is missing, the `bpe` result is saved with an error message instead of crashing the whole script.

## Notes

- These scripts are for debugging and inspection, not formal unit tests.
- Some datasets may trigger download on first run if they are not already present locally.
