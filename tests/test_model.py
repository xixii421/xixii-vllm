import pytest
import torch
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as TransformersQwen3ForCausalLM

from model import Qwen3ForCausalLM


def tiny_config(*, tie_word_embeddings: bool = False) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=1_000_000,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=tie_word_embeddings,
    )
    config._attn_implementation = "eager"
    return config


@torch.no_grad()
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(torch.float32, 1e-5, 1e-5), (torch.bfloat16, 2e-2, 2e-2)],
)
@pytest.mark.parametrize("use_padding_mask", [False, True])
def test_logits_match_transformers_reference(
    use_padding_mask: bool,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    torch.manual_seed(7)
    config = tiny_config()
    reference = TransformersQwen3ForCausalLM(config).eval().to(dtype)
    model = Qwen3ForCausalLM(config).eval().to(dtype)

    # 保留 HF 参数名是 checkpoint 兼容性约定的一部分，
    # 因此必须严格加载，不能静默跳过权重。
    model.load_state_dict(reference.state_dict(), strict=True)
    input_ids = torch.tensor([[11, 23, 5, 42], [0, 0, 8, 9]])
    attention_mask = None
    position_ids = None
    if use_padding_mask:
        attention_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
        position_ids = attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)

    expected = reference(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits
    actual = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@torch.no_grad()
def test_causal_mask_prevents_future_token_leakage() -> None:
    torch.manual_seed(11)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    prefix = torch.tensor([[2, 3, 4]])
    sequence = torch.tensor([[2, 3, 4, 31, 32]])

    prefix_logits = model(prefix)
    sequence_logits = model(sequence)

    torch.testing.assert_close(prefix_logits, sequence_logits[:, :3], rtol=1e-5, atol=1e-5)


def test_tied_embedding_uses_same_storage() -> None:
    model = Qwen3ForCausalLM(tiny_config(tie_word_embeddings=True))
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_logits_to_keep_only_projects_requested_suffix() -> None:
    model = Qwen3ForCausalLM(tiny_config()).eval()
    input_ids = torch.tensor([[2, 3, 4, 5]])

    all_logits = model(input_ids)
    last_logits = model(input_ids, logits_to_keep=1)

    assert last_logits.shape == (1, 1, model.config.vocab_size)
    torch.testing.assert_close(last_logits, all_logits[:, -1:])
