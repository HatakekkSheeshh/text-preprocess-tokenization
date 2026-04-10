# src/evaluation/

Evaluation scripts and metric computation.

Implemented metrics:
- tokenization evaluation:
  - vocabulary size
  - average / median / p95 / max tokens per text
  - unknown-token count and ratio
  - fit time and encode throughput
- loss and perplexity evaluation for LSTM language modeling

Planned additions:
- Perplexity (PP)
- training/inference efficiency

Evaluation outputs should be saved to `outputs/metrics/`.
