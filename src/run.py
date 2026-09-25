import torch

from engine import Engine


def make_generator(
    seed: int | None,
    device: torch.device,
) -> torch.Generator | None:
    """需要可复现性时，创建请求局部的随机数生成器。"""

    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


if __name__ == "__main__":
    engine = Engine.from_config()
    device = next(engine.model.parameters()).device
    generator = make_generator(engine.runtime_config.engine.seed, device)
    output = engine.stream_generate(["如何学习AI"], generator=generator)
    for step,contents in enumerate(output):
        print(f"--------第{step}轮:--------- ",flush=True)
        for id,content in enumerate(contents):
            print(f"第{id}批次内容:  {content}")