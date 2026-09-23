"""读取并校验用户运行配置。

典型用法：

    config = load_config("config.yaml")
    loaded = load_model(
        config.model.name_or_path,
        revision=config.model.revision,
        cache_dir=config.model.cache_dir,
        local_files_only=config.model.local_files_only,
        dtype=config.model.torch_dtype,
        device=config.model.device,
    )

本模块只管理运行参数。模型结构参数由 Hugging Face config.json 管理。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml

from sampler import SamplingParams

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True, slots=True)
class ModelLoadConfig:
    """模型快照位置和运行设备配置。"""

    name_or_path: str
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    dtype: str = "bfloat16"
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if not isinstance(self.name_or_path, str) or not self.name_or_path.strip():
            raise ValueError("model.name_or_path must be a non-empty string")
        if self.revision is not None and not isinstance(self.revision, str):
            raise TypeError("model.revision must be a string or null")
        if self.cache_dir is not None and not isinstance(self.cache_dir, str):
            raise TypeError("model.cache_dir must be a string or null")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("model.local_files_only must be a boolean")
        if not isinstance(self.dtype, str) or self.dtype not in _DTYPE_MAP:
            supported = ", ".join(sorted(_DTYPE_MAP))
            raise ValueError(f"model.dtype must be one of: {supported}")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("model.device must be a non-empty string")
        try:
            torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"invalid model.device: {self.device!r}") from error

    @property
    def torch_dtype(self) -> torch.dtype:
        """返回可直接传给模型加载器的 PyTorch dtype。"""

        return _DTYPE_MAP[self.dtype]


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """v0 单请求生成循环的长度预算和随机种子。"""

    max_model_len: int = 2048
    max_new_tokens: int = 128
    seed: int | None = None

    def __post_init__(self) -> None:
        _require_positive_integer("engine.max_model_len", self.max_model_len)
        _require_positive_integer("engine.max_new_tokens", self.max_new_tokens)
        if self.max_new_tokens > self.max_model_len:
            raise ValueError("engine.max_new_tokens must not exceed max_model_len")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise TypeError("engine.seed must be an integer or null")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Engine 启动所需的完整用户配置。"""

    model: ModelLoadConfig
    engine: EngineConfig
    sampling: SamplingParams


def load_config(path: str | Path = "config.yaml") -> RuntimeConfig:
    """从 YAML 文件读取配置，并拒绝缺失、未知或非法字段。"""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    root = _require_mapping(raw, "config root")
    _reject_unknown_keys(root, {"model", "engine", "sampling"}, "config root")
    if "model" not in root:
        raise ValueError("config must contain a model section")

    model_data = _require_mapping(root["model"], "model")
    engine_data = _require_mapping(root.get("engine", {}), "engine")
    sampling_data = _require_mapping(root.get("sampling", {}), "sampling")

    return RuntimeConfig(
        model=_build_dataclass(ModelLoadConfig, model_data, "model"),
        engine=_build_dataclass(EngineConfig, engine_data, "engine"),
        sampling=_build_dataclass(SamplingParams, sampling_data, "sampling"),
    )


def _require_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_mapping(value: object, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{section} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{section} keys must be strings")
    return value


def _reject_unknown_keys(data: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown keys in {section}: {sorted(unknown)}")


_ConfigType = TypeVar("_ConfigType")


def _build_dataclass(
    config_type: type[_ConfigType],
    data: dict[str, Any],
    section: str,
) -> _ConfigType:
    field_definitions = getattr(config_type, "__dataclass_fields__", None)
    if not isinstance(field_definitions, dict):
        raise TypeError(f"{config_type.__name__} must be a dataclass")
    allowed = set(field_definitions)
    _reject_unknown_keys(data, allowed, section)
    try:
        return config_type(**data)
    except TypeError as error:
        raise TypeError(f"invalid {section} config: {error}") from error
