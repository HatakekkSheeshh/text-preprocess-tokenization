from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass
class LanguageModelingMetrics:
    loss: float
    perplexity: float
    num_sequences: int
    num_target_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_perplexity(loss_value: float) -> float:
    if loss_value >= 20:
        return float("inf")
    return math.exp(loss_value)


@torch.no_grad()
def evaluate_language_model(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> LanguageModelingMetrics:
    model.eval()

    total_loss = 0.0
    total_sequences = 0
    total_target_tokens = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, _ = model(input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        batch_sequences = input_ids.size(0)
        batch_target_tokens = target_ids.numel()
        total_loss += loss.item() * batch_target_tokens
        total_sequences += batch_sequences
        total_target_tokens += batch_target_tokens

    if total_target_tokens == 0:
        raise ValueError("Evaluation dataloader produced zero target tokens.")

    average_loss = total_loss / total_target_tokens
    return LanguageModelingMetrics(
        loss=average_loss,
        perplexity=compute_perplexity(average_loss),
        num_sequences=total_sequences,
        num_target_tokens=total_target_tokens,
    )
