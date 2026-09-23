"""Rotary position embedding operators."""

import torch
from torch import nn
from transformers import Qwen3Config


class RotaryEmbedding(nn.Module):
    """Default (non-scaled) rotary position embedding for Qwen3."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if config.rope_scaling is not None:
            raise NotImplementedError(
                "This v0 implementation supports the Qwen3 default RoPE only "
                "(rope_scaling must be None)."
            )

        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.LongTensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_states.device)[None, :]
        # inv_freq: [D/2], position_ids: [B, S] -> freqs: [B, S, D/2]
        inv_freq = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()
        device_type = hidden_states.device.type
        if device_type == "mps":
            device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float() @ positions).transpose(1, 2)
            angles = torch.cat((freqs, freqs), dim=-1)
            cos = angles.cos()
            sin = angles.sin()
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to tensors shaped ``[B, H, S, D]``."""

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    return query, key
