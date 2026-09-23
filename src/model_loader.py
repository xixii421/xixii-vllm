"""将 Hugging Face Qwen3 checkpoint 加载到本地模型实现。

典型用法：

    loaded = load_model(
        "Qwen/Qwen3-0.6B",
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    model = loaded.model
    hf_config = loaded.config

``name_or_path`` 既可以是 Hugging Face repo id，也可以是已经下载好的本地
模型目录。返回值中的 ``model_dir`` 是实际使用的快照目录，可继续用于加载
同一版本的 tokenizer。该模块只加载配置和权重，不负责文本编码或生成循环。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import Qwen3Config

from model import Qwen3ForCausalLM

_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SAFETENSORS_FILE = "model.safetensors"
_TIED_EMBEDDING_KEYS = frozenset({"model.embed_tokens.weight", "lm_head.weight"})


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """加载完成的本地模型，以及它所使用的 HF 配置和快照目录。"""

    model: Qwen3ForCausalLM
    config: Qwen3Config
    model_dir: Path


def resolve_model_dir(
    name_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """解析本地模型目录，或下载一个固定版本的 HF 快照。"""

    candidate = Path(name_or_path).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.exists():
        raise NotADirectoryError(f"model path is not a directory: {candidate}")

    snapshot_path = snapshot_download(
        repo_id=str(name_or_path),
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.tiktoken",
            "*.txt",
            "*.jinja",
        ],
    )
    return Path(snapshot_path).resolve()


def load_model(
    name_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda:0",
) -> LoadedModel:
    """将 HF safetensors 权重加载到本地 Qwen3ForCausalLM。

    权重分片会逐个复制到 CPU 模型中，以控制主机内存峰值。所有参数名和
    shape 检查完成后，模型才会被移动到指定的 device。

    参数：
        name_or_path：HF repo id 或本地模型目录。
        revision：指定 branch、tag 或 commit；本地目录会忽略该参数。
        cache_dir：Hugging Face 快照缓存目录。
        local_files_only：为 True 时只读取本地缓存，不访问网络。
        dtype：模型最终使用的浮点 dtype。
        device：模型最终所在设备，例如 ``cpu`` 或 ``cuda:0``。

    返回：
        LoadedModel，其中包含已加载模型、HF 配置和实际快照目录。
    """

    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"model dtype must be floating-point, got {dtype}")

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    model_dir = resolve_model_dir(
        name_or_path,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    config = Qwen3Config.from_pretrained(model_dir, local_files_only=True)
    _validate_config(config)

    # 复制权重前先将模型转换为运行时 dtype，避免加载 bf16 checkpoint 时
    # 仍长期保留一份 fp32 模型参数。
    model = Qwen3ForCausalLM(config).to(dtype=dtype)
    weight_files = _find_weight_files(model_dir)
    _load_weight_files(model, weight_files, tied_embeddings=config.tie_word_embeddings)

    if config.tie_word_embeddings:
        # checkpoint 可能保存共享权重的一个或两个名称。加载完成后重新绑定，
        # 明确保证 embedding 和 lm_head 使用同一块参数存储。
        model.lm_head.weight = model.model.embed_tokens.weight

    model.to(device=target_device)
    model.eval()
    return LoadedModel(model=model, config=config, model_dir=model_dir)


def _validate_config(config: Qwen3Config) -> None:
    if config.model_type != "qwen3":
        raise ValueError(f"expected a qwen3 checkpoint, got model_type={config.model_type!r}")
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if config.rope_scaling is not None:
        raise NotImplementedError("the local model does not support rope_scaling yet")


def _find_weight_files(model_dir: Path) -> tuple[Path, ...]:
    index_path = model_dir / _SAFETENSORS_INDEX
    if index_path.is_file():
        index = _read_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path} must contain a non-empty weight_map")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in weight_map.items()
        ):
            raise ValueError(f"{index_path} weight_map must map strings to strings")

        filenames = sorted(set(weight_map.values()))
        files = tuple(_safe_checkpoint_path(model_dir, filename) for filename in filenames)
    else:
        standard_file = model_dir / _SAFETENSORS_FILE
        if standard_file.is_file():
            files = (standard_file,)
        else:
            candidates = tuple(sorted(model_dir.glob("*.safetensors")))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"expected {_SAFETENSORS_FILE} or {_SAFETENSORS_INDEX} in {model_dir}"
                )
            files = candidates

    missing_files = [path for path in files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"checkpoint shards do not exist: {missing_files}")
    return files


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _safe_checkpoint_path(model_dir: Path, filename: str) -> Path:
    path = (model_dir / filename).resolve()
    if not path.is_relative_to(model_dir.resolve()):
        raise ValueError(f"checkpoint shard escapes model directory: {filename!r}")
    return path


def _load_weight_files(
    model: Qwen3ForCausalLM,
    weight_files: tuple[Path, ...],
    *,
    tied_embeddings: bool,
) -> None:
    expected_keys = set(model.state_dict())
    seen_keys: set[str] = set()

    for weight_file in weight_files:
        shard = load_file(str(weight_file), device="cpu")
        shard_keys = set(shard)

        duplicate_keys = seen_keys & shard_keys
        if duplicate_keys:
            raise RuntimeError(
                f"duplicate parameters across checkpoint shards: {sorted(duplicate_keys)}"
            )

        unexpected_keys = shard_keys - expected_keys
        if unexpected_keys:
            raise RuntimeError(
                f"unexpected checkpoint parameters: {sorted(unexpected_keys)}"
            )

        try:
            # 逐分片加载时自然会缺少其他分片中的参数，因此这里暂不检查
            # missing keys；所有分片加载完成后再做一次全局检查。
            model.load_state_dict(shard, strict=False)
        except RuntimeError as error:
            raise RuntimeError(f"failed to load checkpoint shard {weight_file}: {error}") from error
        seen_keys.update(shard_keys)

    missing_keys = expected_keys - seen_keys
    if tied_embeddings:
        present_tied_keys = seen_keys & _TIED_EMBEDDING_KEYS
        if present_tied_keys:
            missing_keys -= _TIED_EMBEDDING_KEYS

    if missing_keys:
        raise RuntimeError(f"missing checkpoint parameters: {sorted(missing_keys)}")
