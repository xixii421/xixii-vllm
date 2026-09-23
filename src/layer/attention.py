"""Grouped-query causal attention operators."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from .norm import RMSNorm
from .rotary import apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand GQA key/value heads from ``H_kv`` to ``H_q``."""

    if repeats == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_kv_heads, repeats, seq_len, head_dim
    )
    return hidden_states.reshape(batch_size, num_kv_heads * repeats, seq_len, head_dim)


def prepare_causal_attention_mask(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Build an additive causal mask shaped ``[B, 1, S, S]``.

    A 2-D mask follows the tokenizer convention (one means a real token, zero
    means padding). A 4-D mask is treated as an already prepared additive mask.
    """

    batch_size, seq_len, _ = hidden_states.shape
    dtype = hidden_states.dtype
    device = hidden_states.device

    if attention_mask is not None and attention_mask.ndim == 4:
        if attention_mask.dtype == torch.bool:
            min_value = torch.finfo(dtype).min
            zeros = torch.zeros(attention_mask.shape, dtype=dtype, device=device)
            return zeros.masked_fill(~attention_mask.to(device=device), min_value)
        return attention_mask.to(device=device, dtype=dtype)

    min_value = torch.finfo(dtype).min
    causal_mask = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device).triu(
        diagonal=1
    )
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

    if attention_mask is None:
        return causal_mask
    if attention_mask.ndim != 2 or attention_mask.shape != (batch_size, seq_len):
        raise ValueError(
            "attention_mask must have shape [batch, sequence] or be a prepared "
            f"4-D mask, got {tuple(attention_mask.shape)}"
        )

    padding = attention_mask[:, None, None, :].to(device=device).eq(0)
    return causal_mask.masked_fill(padding, min_value)


class Qwen3Attention(nn.Module):
    """Qwen3 grouped-query causal self-attention."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # [B, S, H*D] -> [B, H, S, D]
        query = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attention_output)
