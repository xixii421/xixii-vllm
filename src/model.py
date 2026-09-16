class Qwen3Model:
    def __init__(self, config):
        self.config = config

class Qwen3MLP:
    raise NotImplementedError("Qwen3MLP is not implemented yet.")

class Qwen3ForCausalLM:
    self.model : Qwen3Model
    self.lm_head : nn.Linear
    def __init__(self, config):
