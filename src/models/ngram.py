from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class NextTokenPrediction:
    token_id: int
    probability: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SequenceScore:
    num_tokens: int
    log_probability: float
    average_log_probability: float
    average_negative_log_likelihood: float
    perplexity: float

    def to_dict(self) -> dict:
        return asdict(self)


class NGramLanguageModel:
    def __init__(self, *, order: int, vocab_size: int, alpha: float = 1.0) -> None:
        if order <= 0:
            raise ValueError("order must be positive.")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if alpha <= 0:
            raise ValueError("alpha must be positive for Laplace smoothing.")

        self.order = order
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.context_counts: Counter[tuple[int, ...]] = Counter()
        self.next_token_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.num_training_tokens = 0
        self.is_fitted = False

    def fit(self, token_ids: Sequence[int]) -> None:
        self.context_counts.clear()
        self.next_token_counts.clear()
        self.num_training_tokens = len(token_ids)

        for index, token_id in enumerate(token_ids):
            max_context_length = min(self.order - 1, index)
            for context_length in range(max_context_length + 1):
                context = self._build_context(token_ids, index, context_length)
                self.context_counts[context] += 1
                self.next_token_counts[context][token_id] += 1

        self.is_fitted = True

    def score_next_token(self, context_ids: Sequence[int], token_id: int) -> float:
        self._require_fitted()
        context = self._normalize_context(context_ids)
        continuation_counts = self.next_token_counts.get(context)
        observed_count = continuation_counts[token_id] if continuation_counts is not None else 0
        context_count = self.context_counts.get(context, 0)
        numerator = observed_count + self.alpha
        denominator = context_count + (self.alpha * self.vocab_size)
        return numerator / denominator

    def score_sequence(self, token_ids: Sequence[int]) -> SequenceScore:
        self._require_fitted()

        if not token_ids:
            return SequenceScore(
                num_tokens=0,
                log_probability=0.0,
                average_log_probability=0.0,
                average_negative_log_likelihood=0.0,
                perplexity=1.0,
            )

        log_probability = 0.0
        for index, token_id in enumerate(token_ids):
            context = token_ids[max(0, index - (self.order - 1)) : index]
            probability = self.score_next_token(context, token_id)
            log_probability += math.log(probability)

        average_log_probability = log_probability / len(token_ids)
        average_negative_log_likelihood = -log_probability / len(token_ids)
        perplexity = math.exp(average_negative_log_likelihood)

        return SequenceScore(
            num_tokens=len(token_ids),
            log_probability=log_probability,
            average_log_probability=average_log_probability,
            average_negative_log_likelihood=average_negative_log_likelihood,
            perplexity=perplexity,
        )

    def predict_next(
        self,
        context_ids: Sequence[int],
        *,
        top_k: int = 5,
        excluded_token_ids: Sequence[int] | None = None,
    ) -> list[NextTokenPrediction]:
        self._require_fitted()
        excluded = set(excluded_token_ids or [])

        predictions: list[NextTokenPrediction] = []
        for token_id in range(self.vocab_size):
            if token_id in excluded:
                continue
            probability = self.score_next_token(context_ids, token_id)
            predictions.append(NextTokenPrediction(token_id=token_id, probability=probability))

        predictions.sort(key=lambda item: (-item.probability, item.token_id))
        return predictions[:top_k]

    @property
    def num_observed_contexts(self) -> int:
        self._require_fitted()
        return len(self.context_counts)

    @property
    def num_observed_full_order_ngrams(self) -> int:
        self._require_fitted()
        return sum(
            len(continuations)
            for context, continuations in self.next_token_counts.items()
            if len(context) == self.order - 1
        )

    def _normalize_context(self, context_ids: Sequence[int]) -> tuple[int, ...]:
        if self.order == 1:
            return ()
        return tuple(context_ids[-(self.order - 1) :])

    def _build_context(self, token_ids: Sequence[int], end_index: int, context_length: int) -> tuple[int, ...]:
        if context_length == 0:
            return ()
        start_index = end_index - context_length
        return tuple(token_ids[start_index:end_index])

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("The n-gram model must be fitted before scoring or predicting.")
