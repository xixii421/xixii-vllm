from pathlib import Path
from types import SimpleNamespace

import torch

import engine as engine_module
from config import EngineConfig, ModelLoadConfig, RuntimeConfig
from engine import Engine
from sampler import SamplingParams


def test_engine_passes_runtime_model_config_to_loader(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime_config = RuntimeConfig(
        model=ModelLoadConfig(
            name_or_path="local-checkpoint",
            revision="commit-sha",
            cache_dir="/model-cache",
            local_files_only=True,
            dtype="float32",
            device="cpu",
        ),
        engine=EngineConfig(max_model_len=256, max_new_tokens=32, seed=11),
        sampling=SamplingParams(temperature=0.7, top_k=20, top_p=0.8),
    )
    fake_model = object()
    fake_tokenizer = object()
    loader_call: dict[str, object] = {}
    tokenizer_call: dict[str, object] = {}

    def fake_load_model(name_or_path: str, **kwargs):
        loader_call["name_or_path"] = name_or_path
        loader_call.update(kwargs)
        return SimpleNamespace(model=fake_model, model_dir=tmp_path)

    def fake_load_tokenizer(model_dir: Path, **kwargs):
        tokenizer_call["model_dir"] = model_dir
        tokenizer_call.update(kwargs)
        return fake_tokenizer

    monkeypatch.setattr(engine_module, "load_model", fake_load_model)
    monkeypatch.setattr(engine_module.AutoTokenizer, "from_pretrained", fake_load_tokenizer)

    engine = Engine(runtime_config)

    assert loader_call == {
        "name_or_path": "local-checkpoint",
        "revision": "commit-sha",
        "cache_dir": "/model-cache",
        "local_files_only": True,
        "dtype": torch.float32,
        "device": "cpu",
    }
    assert tokenizer_call == {
        "model_dir": tmp_path,
        "local_files_only": True,
        "use_fast": True,
        "trust_remote_code": False,
    }
    assert engine.runtime_config is runtime_config
    assert engine.model is fake_model
    assert engine.tokenizer is fake_tokenizer
    assert engine.runtime_config.sampling.top_k == 20
