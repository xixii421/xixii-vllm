import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as TransformersQwen3ForCausalLM

from model_loader import load_model


def tiny_config(*, tie_word_embeddings: bool = False) -> Qwen3Config:
    return Qwen3Config(
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


def checkpoint_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }


def save_checkpoint(
    model_dir: Path,
    config: Qwen3Config,
    tensors: dict[str, torch.Tensor],
) -> None:
    config.save_pretrained(model_dir)
    save_file(tensors, model_dir / "model.safetensors")


@torch.no_grad()
def test_loads_single_safetensors_into_local_model(tmp_path: Path) -> None:
    torch.manual_seed(17)
    config = tiny_config()
    reference = TransformersQwen3ForCausalLM(config).eval()
    save_checkpoint(tmp_path, config, checkpoint_tensors(reference))

    loaded = load_model(tmp_path, dtype=torch.float32, device="cpu")

    input_ids = torch.tensor([[2, 7, 11, 5]])
    expected = reference(input_ids=input_ids, use_cache=False).logits
    actual = loaded.model(input_ids)

    assert loaded.model_dir == tmp_path.resolve()
    assert loaded.model.training is False
    assert next(loaded.model.parameters()).dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_loads_sharded_safetensors_index(tmp_path: Path) -> None:
    torch.manual_seed(19)
    config = tiny_config()
    reference = TransformersQwen3ForCausalLM(config).eval()
    tensors = checkpoint_tensors(reference)
    config.save_pretrained(tmp_path)

    names = sorted(tensors)
    midpoint = len(names) // 2
    shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    first_names, second_names = names[:midpoint], names[midpoint:]
    save_file({name: tensors[name] for name in first_names}, tmp_path / shard_names[0])
    save_file({name: tensors[name] for name in second_names}, tmp_path / shard_names[1])

    weight_map = {name: shard_names[0] for name in first_names}
    weight_map.update({name: shard_names[1] for name in second_names})
    index = {"metadata": {}, "weight_map": weight_map}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )

    loaded = load_model(tmp_path, dtype=torch.float32, device="cpu")

    for name, expected in tensors.items():
        torch.testing.assert_close(loaded.model.state_dict()[name], expected)


def test_allows_omitted_lm_head_for_tied_embeddings(tmp_path: Path) -> None:
    torch.manual_seed(23)
    config = tiny_config(tie_word_embeddings=True)
    reference = TransformersQwen3ForCausalLM(config).eval()
    tensors = checkpoint_tensors(reference)
    del tensors["lm_head.weight"]
    save_checkpoint(tmp_path, config, tensors)

    loaded = load_model(tmp_path, dtype=torch.float32, device="cpu")

    assert (
        loaded.model.lm_head.weight.data_ptr()
        == loaded.model.model.embed_tokens.weight.data_ptr()
    )
    torch.testing.assert_close(
        loaded.model.model.embed_tokens.weight,
        reference.model.embed_tokens.weight,
    )


def test_rejects_missing_checkpoint_parameter(tmp_path: Path) -> None:
    config = tiny_config()
    reference = TransformersQwen3ForCausalLM(config)
    tensors = checkpoint_tensors(reference)
    missing_name = "model.layers.0.self_attn.q_proj.weight"
    del tensors[missing_name]
    save_checkpoint(tmp_path, config, tensors)

    with pytest.raises(RuntimeError, match=missing_name):
        load_model(tmp_path, dtype=torch.float32, device="cpu")


def test_rejects_unexpected_checkpoint_parameter(tmp_path: Path) -> None:
    config = tiny_config()
    reference = TransformersQwen3ForCausalLM(config)
    tensors = checkpoint_tensors(reference)
    tensors["unexpected.weight"] = torch.ones(1)
    save_checkpoint(tmp_path, config, tensors)

    with pytest.raises(RuntimeError, match="unexpected.weight"):
        load_model(tmp_path, dtype=torch.float32, device="cpu")
