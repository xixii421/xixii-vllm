from copy import deepcopy

import pytest

from benchmarks.benchmark_engine import (
    RunMetrics,
    _format_report,
    _percentile,
    _should_use_color,
    _summarize,
)


def test_percentile_uses_linear_interpolation() -> None:
    assert _percentile([10.0, 20.0, 30.0], 50) == 20.0
    assert _percentile([10.0, 20.0], 95) == pytest.approx(19.5)


def test_summary_aggregates_latency_throughput_and_memory() -> None:
    runs = [
        RunMetrics(2, 2, 30.0, 10.0, (20.0,), 66.0, 1.2, 0.2),
        RunMetrics(2, 2, 40.0, 20.0, (20.0,), 50.0, 1.3, 0.3),
    ]

    summary = _summarize(runs)

    assert summary["ttft_ms"]["p50"] == 15.0
    assert summary["itl_ms"]["p50"] == 20.0
    assert summary["throughput_token_slots_per_s"]["p50"] == 58.0
    assert summary["peak_allocated_gib_max"] == 1.3
    assert summary["peak_incremental_gib_max"] == 0.3


def _sample_report() -> dict:
    return {
        "timestamp_utc": "2026-09-25T08:00:00+00:00",
        "git": {"commit": "1234567890abcdef", "dirty": True},
        "environment": {
            "python": "3.12.13",
            "torch": "2.13.0",
            "cuda_runtime": "13.0",
            "device": "cuda:0",
            "gpu": "NVIDIA RTX 4060",
        },
        "runtime_config": {
            "model": {"name_or_path": "Qwen/Qwen3-0.6B", "dtype": "bfloat16"},
            "sampling": {"temperature": 0.9, "top_k": 50, "top_p": 0.9},
        },
        "workload": {
            "batch_size": 2,
            "prompt_tokens_per_request": 16,
            "max_new_tokens": 32,
            "seed": 7,
            "warmup": 2,
            "repeats": 5,
        },
        "aggregate": {
            "generated_steps": [30, 32],
            "total_latency_ms": {"p50": 120.0, "p95": 140.0},
            "ttft_ms": {"p50": 20.0, "p95": 25.0},
            "itl_ms": {"p50": 3.2, "p95": 4.5},
            "throughput_token_slots_per_s": {"p50": 500.0, "p95": 520.0},
            "peak_allocated_gib_max": 1.3,
            "peak_incremental_gib_max": 0.3,
        },
    }


def test_format_report_groups_key_results_for_terminal_reading() -> None:
    output = _format_report(_sample_report())

    assert "工作负载" in output
    assert "性能摘要" in output
    assert "复现信息" in output
    assert "NVIDIA RTX 4060 (cuda:0)" in output
    assert "30–32" in output
    assert "120.00" in output
    assert "500.00" in output
    assert "12345678（工作区有未提交改动）" in output


def test_format_report_marks_unavailable_gpu_metrics() -> None:
    report = deepcopy(_sample_report())
    report["environment"] = {
        "python": "3.12.13",
        "torch": "2.13.0",
        "cuda_runtime": None,
        "device": "cpu",
    }
    report["workload"]["seed"] = None
    report["aggregate"]["itl_ms"] = None
    report["aggregate"]["peak_allocated_gib_max"] = None
    report["aggregate"]["peak_incremental_gib_max"] = None

    output = _format_report(report)

    assert "Token 间延迟 (ITL)" in output
    assert "—" in output
    assert output.count("不可用（仅 CUDA）") == 2
    assert "未设置" in output


def test_format_report_adds_ansi_styles_only_when_enabled() -> None:
    plain = _format_report(_sample_report())
    colored = _format_report(_sample_report(), color=True)

    escape = f"{chr(27)}["
    assert escape not in plain
    assert f"{escape}1;36m" in colored
    assert f"{escape}32m" in colored
    assert f"{escape}35m" in colored
    assert "120.00" in colored


@pytest.mark.parametrize(
    ("mode", "is_tty", "no_color", "expected"),
    [
        ("auto", True, False, True),
        ("auto", False, False, False),
        ("auto", True, True, False),
        ("always", False, True, True),
        ("never", True, False, False),
    ],
)
def test_color_policy(
    mode: str,
    is_tty: bool,
    no_color: bool,
    expected: bool,
) -> None:
    assert _should_use_color(mode, is_tty=is_tty, no_color=no_color) is expected
