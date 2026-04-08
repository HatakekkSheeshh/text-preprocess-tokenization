from __future__ import annotations

import torch
from torch import nn


class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_token_id: int | None = None,
    ) -> None:
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id,
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)

        outputs, hidden_state = self.lstm(embedded, hidden_state)
        outputs = self.output_dropout(outputs)
        logits = self.output_layer(outputs)
        return logits, hidden_state

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
