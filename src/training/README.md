# src/training/

Training scripts and training utilities.

Typical responsibilities:
- build dataloaders
- run the training loop
- save checkpoints to `outputs/checkpoints/`
- log training metrics

Currently implemented:
- a baseline next-token prediction training pipeline for `LSTMLanguageModel`
- reusable LM dataset slicing for contiguous token streams
- checkpoint and metric export for later report analysis
