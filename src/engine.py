from collections.abc import Iterator
from pathlib import Path
from typing import Self

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import RuntimeConfig, load_config
from model import Qwen3ForCausalLM
from model_loader import load_model
from sampler import Sampler
from dataclasses import dataclass
@dataclass(frozen = True,slots = True)
class BatchGenerationStep:
    step  : int
    request_ids : tuple[str]
    next_token_ids : tuple[int | None,...]
    current_text:tuple[str|None,...]
    finished : tuple[bool]
    finished_reasons :tuple[str|None]

class Engine:
    model: Qwen3ForCausalLM
    sampler: Sampler
    tokenizer: PreTrainedTokenizerBase
    runtime_config: RuntimeConfig

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self.runtime_config = runtime_config
        model_config = runtime_config.model
        loaded = load_model(
            model_config.name_or_path,
            revision=model_config.revision,
            cache_dir=model_config.cache_dir,
            local_files_only=model_config.local_files_only,
            dtype=model_config.torch_dtype,
            device=model_config.device,
        )
        self.model = loaded.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            loaded.model_dir,
            local_files_only=True,
            use_fast=True,
            trust_remote_code=False,
        )
        self.sampler = Sampler()

    @classmethod
    def from_config(cls, path: str | Path = "config.yaml") -> Self:
        """读取 YAML 运行配置并创建 Engine。"""

        return cls(load_config(path))

    def stream_generate(self, prompts: list[str]) -> Iterator[BatchGenerationStep]:
        for step in range(self.runtime_config.engine.max_new_tokens):
            self.tokenizer()
            raise NotImplementedError
        
