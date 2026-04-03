# src/datasets/

Code for loading datasets and applying text preprocessing.

Typical responsibilities:
- load datasets from Hugging Face, Kaggle files, or local paths
- normalize/clean raw text
- create train/validation/test splits
- export processed data to `data/processed/`

## WikiText-103
Due to the assignment requirements, we focus mainly on tokenization and avoid fine-grained preprocessing as much as possible to save time. Therefore, we choose wikitext-103-v1 instead of wikitext-103-raw-v1 from source: [WikiText Dataset](https://huggingface.co/datasets/Salesforce/wikitext)

In this assignment, we ultilize **datasets** library to load data into our work:
```python
from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
```
