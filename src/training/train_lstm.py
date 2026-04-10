from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.language_modeling import evaluate_language_model
from src.models.lstm_language_model import LSTMLanguageModel
from src.tokenizers import build_tokenizer
from src.training.lm_data import LanguageModelingDataset, build_prepared_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints" / "lstm"
METRICS_ROOT = PROJECT_ROOT / "outputs" / "metrics" / "lstm"


@dataclass
class LSTMTrainingConfig:
    dataset_name: str
    tokenizer_name: str = "word"
    sequence_length: int = 128
    stride: int | None = None
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0
    min_freq: int = 1
    max_vocab_size: int | None = 50_000
    max_fit_texts: int | None = None
    max_train_tokens: int | None = None
    max_validation_tokens: int | None = None
    max_test_tokens: int | None = None
    device: str | None = None
    run_name: str | None = None
    log_interval: int = 100


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_name(config: LSTMTrainingConfig) -> str:
    if config.run_name:
        return config.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"lstm_{config.dataset_name}_{config.tokenizer_name}_{timestamp}"


def build_dataloaders(prepared_corpus, config: LSTMTrainingConfig) -> dict[str, DataLoader]:
    dataloaders: dict[str, DataLoader] = {}

    for split_name, prepared_split in prepared_corpus.splits.items():
        dataset = LanguageModelingDataset(
            prepared_split.token_ids,
            sequence_length=config.sequence_length,
            stride=config.stride,
        )
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split_name == "train",
            num_workers=config.num_workers,
        )

    return dataloaders


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: float | None = None,
    log_interval: int = 100,
) -> dict[str, float]:
    model.train()

    running_loss = 0.0
    total_loss = 0.0
    total_target_tokens = 0
    total_sequences = 0
    start_time = time.perf_counter()

    for step, (input_ids, target_ids) in enumerate(dataloader, start=1):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_target_tokens = target_ids.numel()
        batch_sequences = input_ids.size(0)
        total_loss += loss.item() * batch_target_tokens
        running_loss += loss.item()
        total_target_tokens += batch_target_tokens
        total_sequences += batch_sequences

        if step % log_interval == 0:
            print(
                f"  step {step:04d} | "
                f"avg loss {running_loss / log_interval:.4f}"
            )
            running_loss = 0.0

    duration = time.perf_counter() - start_time
    average_loss = total_loss / total_target_tokens
    tokens_per_second = total_target_tokens / duration if duration > 0 else 0.0

    return {
        "loss": average_loss,
        "perplexity": float("inf") if average_loss >= 20 else float(torch.exp(torch.tensor(average_loss))),
        "num_sequences": total_sequences,
        "num_target_tokens": total_target_tokens,
        "duration_seconds": duration,
        "tokens_per_second": tokens_per_second,
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(
    path: Path,
    *,
    model: LSTMLanguageModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: LSTMTrainingConfig,
    tokenizer_vocab_size: int,
    tokenizer_type: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "vocab_size": tokenizer_vocab_size,
            "tokenizer_type": tokenizer_type,
        },
        path,
    )


def train_lstm_language_model(config: LSTMTrainingConfig) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_name = build_run_name(config)

    tokenizer = build_tokenizer(
        config.tokenizer_name,
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
    )
    prepared_corpus = build_prepared_corpus(
        config.dataset_name,
        tokenizer,
        max_fit_texts=config.max_fit_texts,
        max_train_tokens=config.max_train_tokens,
        max_validation_tokens=config.max_validation_tokens,
        max_test_tokens=config.max_test_tokens,
        sequence_length=config.sequence_length,
        stride=config.stride,
    )
    dataloaders = build_dataloaders(prepared_corpus, config)

    model = LSTMLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = CHECKPOINT_ROOT / run_name
    metrics_path = METRICS_ROOT / f"{run_name}.json"
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    print(
        "Prepared LSTM run with "
        f"dataset={config.dataset_name}, tokenizer={config.tokenizer_name}, "
        f"vocab_size={tokenizer.vocab_size}, device={device.type}"
    )

    history: list[dict] = []
    best_validation_loss = float("inf")
    total_training_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        train_metrics = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device,
            grad_clip=config.grad_clip,
            log_interval=config.log_interval,
        )
        validation_metrics = evaluate_language_model(
            model,
            dataloaders["validation"],
            criterion,
            device,
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics.to_dict(),
        }
        history.append(epoch_record)

        print(
            f"  train loss {train_metrics['loss']:.4f} | "
            f"train ppl {train_metrics['perplexity']:.2f} | "
            f"val loss {validation_metrics.loss:.4f} | "
            f"val ppl {validation_metrics.perplexity:.2f}"
        )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            tokenizer_vocab_size=tokenizer.vocab_size,
            tokenizer_type=tokenizer.tokenizer_type,
        )

        if validation_metrics.loss < best_validation_loss:
            best_validation_loss = validation_metrics.loss
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                tokenizer_vocab_size=tokenizer.vocab_size,
                tokenizer_type=tokenizer.tokenizer_type,
            )

    total_training_seconds = time.perf_counter() - total_training_start
    test_metrics = evaluate_language_model(
        model,
        dataloaders["test"],
        criterion,
        device,
    )

    summary = {
        "run_name": run_name,
        "config": {
            **asdict(config),
            "device": str(device),
        },
        "model": {
            "type": "lstm-language-model",
            "num_parameters": model.num_parameters,
        },
        "tokenizer": {
            "type": tokenizer.tokenizer_type,
            "vocab_size": tokenizer.vocab_size,
        },
        "dataset": {
            split_name: {
                "num_tokens": prepared_split.num_tokens,
                "num_sequences": prepared_split.num_sequences,
            }
            for split_name, prepared_split in prepared_corpus.splits.items()
        },
        "history": history,
        "best_validation_loss": best_validation_loss,
        "test": test_metrics.to_dict(),
        "total_training_seconds": total_training_seconds,
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer_path": str(tokenizer_path),
    }

    save_json(metrics_path, summary)
    print(f"Saved checkpoints to {checkpoint_dir}")
    print(f"Saved metrics to {metrics_path}")
    return summary
