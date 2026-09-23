"""Feed-forward operators."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config


class Qwen3MLP(nn.Module):
    """Qwen3 SwiGLU feed-forward operator."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Qwen3 expects hidden_act='silu', got {config.hidden_act!r}")
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
