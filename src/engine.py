from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import RuntimeConfig, load_config
from model import Qwen3ForCausalLM
from model_loader import load_model
from sampler import Sampler


@dataclass(frozen = True,slots = True)
class BatchGenerationStep:
    step  : int
    request_ids : tuple[str,...]
    next_token_ids : tuple[int | None,...]
    current_text:tuple[str|None,...]
    finished : tuple[bool,...]
    finished_reasons :tuple[str|None,...]

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
        self.tokenizer.padding_side = "left"
        self.sampler = Sampler()

    @classmethod
    def from_config(cls, path: str | Path = "config.yaml") -> Self:
        """读取 YAML 运行配置并创建 Engine。"""

        return cls(load_config(path))

    def stream_generate(
        self,
        prompts: list[str],
        *,
        generator: torch.Generator | None = None,
    ) -> Iterator[list[str]]:
        inputs = self.tokenizer(prompts,padding=True,return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(device) # pyright: ignore[reportAttributeAccessIssue]
        attention_mask = inputs["attention_mask"]
        config = self.runtime_config.engine
        batch_size = len(prompts)
        cur_token_ids = [[] for _ in range(batch_size)]
        cur_texts : list[str] = ["" for _ in range(batch_size)]
        finished  = [False  for _ in range(batch_size)]
        for step in range(config.max_new_tokens):
            output_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=1,
            )
            output_ids = self.sampler(
                output_logits,
                self.runtime_config.sampling,
                generator=generator,
            )
            next_token_mask = torch.tensor([False for  _ in range(batch_size)]).unsqueeze(dim= -1)
            for id,token_id in enumerate(output_ids):
                if finished[id]:
                    output_ids[id] = self.tokenizer.pad_token_id
                    next_token_mask[id][0] = False
                    continue
                if token_id == self.tokenizer.eos_token_id or input_ids.shape[1] + 1 == config.max_model_len:
                    finished[id] =  True
                    next_token_mask[id][0] = False

                cur_token_ids[id].append(token_id)
                cur_texts[id] = self.tokenizer.decode(cur_token_ids[id])
                next_token_mask[id][0] = True
            input_ids = torch.cat([input_ids,output_ids.unsqueeze(dim = 1)],dim = 1)
            attention_mask = torch.cat([attention_mask,next_token_mask],dim=1) # pyright: ignore[reportArgumentType]
            yield cur_texts.copy()
            if all(finished):
                break