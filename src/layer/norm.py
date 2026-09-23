"""Normalization operators."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by Qwen3."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        states_fp32 = hidden_states.float()
        variance = states_fp32.square().mean(dim=-1, keepdim=True)
        normalized = states_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


Qwen3RMSNorm = RMSNorm
