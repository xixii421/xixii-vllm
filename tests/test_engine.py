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
    fake_tokenizer = SimpleNamespace(padding_side="right")
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
    assert engine.tokenizer.padding_side == "left"
    assert engine.runtime_config.sampling.top_k == 20


def test_stream_generate_forwards_request_generator_to_sampler() -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.empty(0))

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            logits_to_keep: int,
        ) -> torch.Tensor:
            assert logits_to_keep == 1
            batch_size = input_ids.shape[0]
            return torch.zeros(batch_size, input_ids.shape[1], 4)

    class FakeTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, prompts, **kwargs):
            batch_size = len(prompts)
            return {
                "input_ids": torch.ones(batch_size, 1, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, 1, dtype=torch.long),
            }

        def decode(self, token_ids) -> str:
            return "done"

    received_generator: torch.Generator | None = None

    def fake_sampler(
        logits: torch.Tensor,
        params: SamplingParams,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        nonlocal received_generator
        received_generator = generator
        return torch.full((logits.shape[0],), 2, dtype=torch.long)

    runtime_config = RuntimeConfig(
        model=ModelLoadConfig(name_or_path="unused", device="cpu"),
        engine=EngineConfig(max_model_len=8, max_new_tokens=1, seed=7),
        sampling=SamplingParams(),
    )
    engine = Engine.__new__(Engine)
    engine.runtime_config = runtime_config
    engine.model = FakeModel()
    engine.tokenizer = FakeTokenizer()
    engine.sampler = fake_sampler
    generator = torch.Generator().manual_seed(7)

    list(engine.stream_generate(["prompt"], generator=generator))

    assert received_generator is generator
