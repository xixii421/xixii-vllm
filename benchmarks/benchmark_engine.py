"""对 eager Engine 流式生成路径进行 benchmark。

在仓库根目录中运行：

    PYTHONPATH=src uv run python benchmarks/benchmark_engine.py \
        --max-new-tokens 32 --warmup 2 --repeats 5

计时不包含模型加载，但包含 tokenization、模型执行、采样、detokenization 和流式输出。
批次大小大于 1 时，由于当前 Engine 流不暴露单个请求的完成状态，吞吐量使用
“已处理 token slot”作为单位。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from config import RuntimeConfig, load_config
from engine import Engine


@dataclass(frozen=True, slots=True)
class RunMetrics:
    generated_steps: int
    processed_token_slots: int
    total_latency_ms: float
    ttft_ms: float
    inter_token_latencies_ms: tuple[float, ...]
    throughput_token_slots_per_s: float
    peak_allocated_gib: float | None
    peak_incremental_gib: float | None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _prompt_token_count(engine: Engine, prompt: str) -> int:
    encoded = engine.tokenizer([prompt], padding=True, return_tensors="pt")
    attention_mask = encoded["attention_mask"]
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("tokenizer attention_mask must be a Tensor")
    return int(attention_mask[0].sum().item())


def _measure_once(
    engine: Engine,
    prompts: list[str],
    *,
    seed: int | None,
) -> RunMetrics:
    device = next(engine.model.parameters()).device
    generator = _make_generator(seed, device)

    _synchronize(device)
    baseline_allocated = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)

    token_times: list[float] = []
    start = perf_counter()
    with torch.inference_mode():
        for _ in engine.stream_generate(prompts, generator=generator):
            # 只有已排队的 CUDA 工作完成后，客户端才能观测到 yield 的 token，
            # 因此在每个流式输出边界执行同步。
            _synchronize(device)
            token_times.append(perf_counter())
    _synchronize(device)
    end = perf_counter()

    if not token_times:
        raise RuntimeError("Engine produced no generation steps")

    inter_token_latencies_ms = tuple(
        (current - previous) * 1000 for previous, current in zip(token_times, token_times[1:])
    )
    total_latency_s = end - start
    generated_steps = len(token_times)
    processed_token_slots = generated_steps * len(prompts)

    peak_allocated_gib: float | None = None
    peak_incremental_gib: float | None = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_allocated_gib = peak_allocated / 1024**3
        peak_incremental_gib = max(0, peak_allocated - baseline_allocated) / 1024**3

    return RunMetrics(
        generated_steps=generated_steps,
        processed_token_slots=processed_token_slots,
        total_latency_ms=total_latency_s * 1000,
        ttft_ms=(token_times[0] - start) * 1000,
        inter_token_latencies_ms=inter_token_latencies_ms,
        throughput_token_slots_per_s=processed_token_slots / total_latency_s,
        peak_allocated_gib=peak_allocated_gib,
        peak_incremental_gib=peak_incremental_gib,
    )


def _summarize(runs: list[RunMetrics]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one measured run is required")

    ttft = [run.ttft_ms for run in runs]
    total_latency = [run.total_latency_ms for run in runs]
    throughput = [run.throughput_token_slots_per_s for run in runs]
    itl = [latency for run in runs for latency in run.inter_token_latencies_ms]
    peak_allocated = [run.peak_allocated_gib for run in runs if run.peak_allocated_gib is not None]
    peak_incremental = [
        run.peak_incremental_gib for run in runs if run.peak_incremental_gib is not None
    ]

    return {
        "repeats": len(runs),
        "generated_steps": [run.generated_steps for run in runs],
        "processed_token_slots": [run.processed_token_slots for run in runs],
        "total_latency_ms": {
            "p50": _percentile(total_latency, 50),
            "p95": _percentile(total_latency, 95),
        },
        "ttft_ms": {
            "p50": _percentile(ttft, 50),
            "p95": _percentile(ttft, 95),
        },
        "itl_ms": ({"p50": _percentile(itl, 50), "p95": _percentile(itl, 95)} if itl else None),
        "throughput_token_slots_per_s": {
            "p50": _percentile(throughput, 50),
            "p95": _percentile(throughput, 95),
        },
        "peak_allocated_gib_max": max(peak_allocated) if peak_allocated else None,
        "peak_incremental_gib_max": max(peak_incremental) if peak_incremental else None,
    }


_ANSI_STYLES = {
    "bold": "1",
    "dim": "2",
    "blue": "34",
    "cyan": "36",
    "green": "32",
    "magenta": "35",
}


def _colorize(text: str, *styles: str, enabled: bool) -> str:
    """按需为终端文本添加 ANSI 样式。"""

    if not enabled:
        return text
    codes = ";".join(_ANSI_STYLES[style] for style in styles)
    escape = chr(27)
    return f"{escape}[{codes}m{text}{escape}[0m"


def _should_use_color(mode: str, *, is_tty: bool, no_color: bool) -> bool:
    """根据命令行模式和终端环境判断是否启用颜色。"""

    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode not in {"auto"}:
        raise ValueError(f"unknown color mode: {mode}")
    return is_tty and not no_color


def _display_width(value: str) -> int:
    """返回字符串在等宽终端中的大致显示宽度。"""

    return sum(
        2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1 for character in value
    )


def _pad(value: object, width: int, *, align: str = "left") -> str:
    """按终端显示宽度填充文本。"""

    text = str(value)
    padding = " " * max(0, width - _display_width(text))
    return f"{padding}{text}" if align == "right" else f"{text}{padding}"


def _format_number(value: float | None) -> str:
    return "—" if value is None else f"{value:,.2f}"


def _format_step_range(steps: list[int]) -> str:
    if not steps:
        return "—"
    minimum = min(steps)
    maximum = max(steps)
    return str(minimum) if minimum == maximum else f"{minimum}–{maximum}"


def _format_report(report: dict[str, Any], *, color: bool = False) -> str:
    """将完整 benchmark 报告格式化为适合终端阅读的摘要。"""

    environment = report["environment"]
    runtime_config = report["runtime_config"]
    workload = report["workload"]
    aggregate = report["aggregate"]

    model_config = runtime_config["model"]
    sampling_config = runtime_config["sampling"]
    gpu = environment.get("gpu")
    device = environment["device"]
    device_label = f"{gpu} ({device})" if gpu else device

    git = report["git"]
    commit = git.get("commit")
    git_label = commit[:8] if commit else "不可用"
    if git.get("dirty"):
        git_label += "（工作区有未提交改动）"

    separator = _colorize("─" * 72, "dim", "cyan", enabled=color)
    label_width = 24
    value_width = 14

    def section(title: str) -> str:
        return _colorize(title, "bold", "cyan", enabled=color)

    def detail(label: str, value: object) -> str:
        label_cell = _colorize(_pad(label, label_width), "blue", enabled=color)
        return f"  {label_cell}{value}"

    def metric(label: str, values: dict[str, float] | None, unit: str) -> str:
        p50 = _format_number(values["p50"] if values is not None else None)
        p95 = _format_number(values["p95"] if values is not None else None)
        label_cell = _colorize(_pad(label, label_width), "blue", enabled=color)
        p50_cell = _colorize(_pad(p50, value_width, align="right"), "green", enabled=color)
        p95_cell = _colorize(_pad(p95, value_width, align="right"), "magenta", enabled=color)
        unit_cell = _colorize(unit, "dim", enabled=color)
        return f"  {label_cell}{p50_cell}{p95_cell}  {unit_cell}"

    sampling_label = (
        f"temperature={sampling_config['temperature']}, "
        f"top_k={sampling_config['top_k']}, top_p={sampling_config['top_p']}"
    )
    metric_header = (
        f"  {_colorize(_pad('指标', label_width), 'bold', enabled=color)}"
        f"{_colorize(_pad('P50', value_width, align='right'), 'bold', 'green', enabled=color)}"
        f"{_colorize(_pad('P95', value_width, align='right'), 'bold', 'magenta', enabled=color)}"
        f"  {_colorize('单位', 'bold', enabled=color)}"
    )
    metric_rule_text = (
        f"  {'─' * label_width}  {'─' * (value_width - 2)}  {'─' * (value_width - 2)}  {'─' * 12}"
    )
    metric_rule = _colorize(metric_rule_text, "dim", enabled=color)

    peak_allocated = aggregate["peak_allocated_gib_max"]
    peak_incremental = aggregate["peak_incremental_gib_max"]
    peak_allocated_label = (
        f"{_format_number(peak_allocated)} GiB"
        if peak_allocated is not None
        else "不可用（仅 CUDA）"
    )
    peak_incremental_label = (
        f"{_format_number(peak_incremental)} GiB"
        if peak_incremental is not None
        else "不可用（仅 CUDA）"
    )

    lines = [
        separator,
        _colorize("xixii-vllm Benchmark 报告", "bold", "cyan", enabled=color),
        separator,
        "",
        section("工作负载"),
        detail("模型", model_config["name_or_path"]),
        detail("设备", device_label),
        detail("dtype", model_config["dtype"]),
        detail("批次大小", workload["batch_size"]),
        detail("每请求 prompt token", workload["prompt_tokens_per_request"]),
        detail("每轮生成步数", _format_step_range(aggregate["generated_steps"])),
        detail("最大新增 token", workload["max_new_tokens"]),
        detail("采样参数", sampling_label),
        "",
        section("性能摘要"),
        metric_header,
        metric_rule,
        metric("总延迟", aggregate["total_latency_ms"], "ms"),
        metric("首 token 延迟 (TTFT)", aggregate["ttft_ms"], "ms"),
        metric("Token 间延迟 (ITL)", aggregate["itl_ms"], "ms"),
        metric("吞吐量", aggregate["throughput_token_slots_per_s"], "token slot/s"),
        "",
        section("显存"),
        detail("峰值已分配", peak_allocated_label),
        detail("峰值增量", peak_incremental_label),
        "",
        section("复现信息"),
        detail("预热 / 测量轮数", f"{workload['warmup']} / {workload['repeats']}"),
        detail("随机种子", workload["seed"] if workload["seed"] is not None else "未设置"),
        detail("Git", git_label),
        detail(
            "Python / PyTorch / CUDA",
            f"{environment['python']} / {environment['torch']} / "
            f"{environment.get('cuda_runtime') or '不可用'}",
        ),
        detail("时间 (UTC)", report["timestamp_utc"]),
        "",
        _colorize(
            "  说明：吞吐量单位为已处理 token slot/s；batch_size=1 时等同于生成 token/s。",
            "dim",
            enabled=color,
        ),
        separator,
    ]
    return chr(10).join(lines)


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        metadata.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "gpu_total_memory_gib": (
                    torch.cuda.get_device_properties(device).total_memory / 1024**3
                ),
            }
        )
    return metadata


def _build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    runtime_config = load_config(args.config)
    if args.max_new_tokens is not None:
        runtime_config = replace(
            runtime_config,
            engine=replace(runtime_config.engine, max_new_tokens=args.max_new_tokens),
        )
    return runtime_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--prompt", default="请简要解释什么是 KV Cache。")
    parser.add_argument(
        "--prompt-repeat",
        type=_positive_int,
        default=1,
        help="重复 prompt 以构造更长输入；报告会记录实际 token 数。",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--max-new-tokens", type=_positive_int)
    parser.add_argument("--warmup", type=_positive_int, default=2)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--seed", type=int, help="覆盖配置文件中的 engine.seed。")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true", help="在标准输出中打印完整 JSON 报告。")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="控制终端报告的 ANSI 配色。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime_config = _build_runtime_config(args)
    seed = args.seed if args.seed is not None else runtime_config.engine.seed
    prompt = " ".join([args.prompt] * args.prompt_repeat)
    prompts = [prompt] * args.batch_size

    print("[1/3] 正在加载模型（不计入 benchmark 时间）...", file=sys.stderr, flush=True)
    engine = Engine(runtime_config)
    device = next(engine.model.parameters()).device
    prompt_tokens = _prompt_token_count(engine, prompt)

    for index in range(args.warmup):
        print(f"[2/3] 预热 {index + 1}/{args.warmup}", file=sys.stderr, flush=True)
        _measure_once(engine, prompts, seed=seed)

    runs: list[RunMetrics] = []
    for index in range(args.repeats):
        print(f"[3/3] 测量 {index + 1}/{args.repeats}", file=sys.stderr, flush=True)
        runs.append(_measure_once(engine, prompts, seed=seed))

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git": _git_metadata(),
        "environment": _environment_metadata(device),
        "runtime_config": asdict(runtime_config),
        "workload": {
            "input_format": "raw completion",
            "batch_size": args.batch_size,
            "prompt_characters_per_request": len(prompt),
            "prompt_tokens_per_request": prompt_tokens,
            "max_new_tokens": runtime_config.engine.max_new_tokens,
            "seed": seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "timing_includes": [
                "tokenization",
                "model_forward",
                "sampling",
                "detokenization",
                "streaming",
            ],
            "timing_excludes": ["model_loading"],
            "throughput_unit": "processed token slots; equals generated tokens for batch_size=1",
        },
        "aggregate": _summarize(runs),
        "runs": [asdict(run) for run in runs],
    }

    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    use_color = _should_use_color(
        args.color,
        is_tty=sys.stdout.isatty(),
        no_color=bool(os.environ.get("NO_COLOR")),
    )
    print(serialized if args.json else _format_report(report, color=use_color))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + chr(10), encoding="utf-8")
        print(f"JSON 报告已保存至：{args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
