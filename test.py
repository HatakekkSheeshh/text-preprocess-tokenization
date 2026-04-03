from datasets import load_from_disk

ds = load_from_disk("data/raw/wikitext-103")
print(ds["train"]["text"][1:5])