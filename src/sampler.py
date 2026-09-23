"""Token sampling for the eager v0 generation path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Parameters shared by one sampling batch.

    ``temperature=0`` selects greedy decoding. ``top_k=0`` and ``top_p=1``
    disable their respective filters.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be finite and in the interval (0, 1]")


class Sampler(nn.Module):
    """Select one next-token id per batch row from model logits.

    The preferred input shape is ``[B, V]``. For convenience, logits shaped
    ``[B, S, V]`` are also accepted and only the last sequence position is
    sampled. Sampling is performed in fp32 for fp16/bf16 logits.
    """

    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        params: SamplingParams | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.LongTensor:
        params = params or SamplingParams()
        next_token_logits = _select_next_token_logits(logits)
        if not torch.isfinite(next_token_logits).all():
            raise ValueError("logits must contain only finite values")

        if params.temperature == 0:
            return next_token_logits.argmax(dim=-1)

        scores = next_token_logits.float().div(params.temperature)
        scores = _apply_top_k(scores, params.top_k)
        scores = _apply_top_p(scores, params.top_p)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)


def _select_next_token_logits(logits: torch.Tensor) -> torch.Tensor:
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if logits.ndim == 3:
        if logits.shape[1] == 0:
            raise ValueError("sequence dimension must not be empty")
        logits = logits[:, -1, :]
    elif logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, V] or [B, S, V], got {tuple(logits.shape)}")
    if logits.shape[0] == 0:
        raise ValueError("batch dimension must not be empty")
    if logits.shape[1] == 0:
        raise ValueError("vocabulary dimension must not be empty")
    return logits


def _apply_top_k(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k == 0 or top_k >= scores.shape[-1]:
        return scores

    top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
    filtered = torch.full_like(scores, -torch.inf)
    return filtered.scatter(dim=-1, index=top_indices, src=top_scores)


def _apply_top_p(scores: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p == 1:
        return scores

    sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
    cumulative_probabilities = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)

    # Shift the mask so the first token crossing top_p remains eligible. This
    # keeps the smallest high-probability prefix whose mass reaches top_p.
    sorted_remove = cumulative_probabilities > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    remove = torch.zeros_like(sorted_remove).scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_remove,
    )
    return scores.masked_fill(remove, -torch.inf)
