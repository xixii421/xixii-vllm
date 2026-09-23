# xixii-vllm

一个以学习和求职展示为目标、从零实现的单卡 LLM 推理引擎。

项目参考隔壁 `dzyy-vllm` 的能力演进，但不会复制它的源码。这里采用一条持续演进的
`src/` 主线，并在阶段验收后使用 Git tag 保留 `v0`、`v1`、`v2`、`v3` 里程碑。

> 当前状态：项目初始化。推理引擎尚未实现，阶段复选框代表真实完成状态。

## 一个月目标

每周投入约 10 小时，在四周内尽量达到 `dzyy-vllm/v3` 的能力边界：

| 周次 | 里程碑 | 核心问题 | 可展示交付物 |
| --- | --- | --- | --- |
| 第 1 周 | v0：正确性基线 | Qwen3 前向、权重加载、采样与自回归循环 | 与 Hugging Face 对齐的测试、可运行生成示例 |
| 第 2 周 | v1：KV 内存系统 | KV Cache、物理块、页表、slot mapping、前缀复用 | 生命周期不变量测试、显存预算与复用实验 |
| 第 3 周 | v2：请求调度 | Continuous Batching、chunked prefill、prefill/decode 混跑 | 调度时间线测试、并发吞吐与延迟基线 |
| 第 4 周 | v3：性能优化 | Paged Attention、批量 decode、CUDA Graph、`torch.compile` | profiler 证据、优化前后 benchmark、瓶颈复盘 |

40 小时是进取预算。阶段完成以验收证据为准，而不是以日历或“代码已经写过”为准。
FlashAttention、CUDA Graph 和 `torch.compile` 中至少完成一条有完整性能证据的优化链；
其余未完成项应如实记录为后续工作。

## 真正要学习的东西

这个项目不以“复刻一个能跑的仓库”为终点，而是训练四种能力：

1. **架构设计**：从请求生命周期和数据流出发，划分控制面、数据面与计算面，并说清模块边界。
2. **机制理解**：通过亲手实现 Qwen3、KV Cache、分页映射和调度器，掌握 shape、状态和不变量。
3. **优化判断**：先定位瓶颈，再提出假设和选择优化手段；不把使用高性能库等同于理解性能。
4. **性能分析**：区分吞吐、首 token 延迟、逐 token 延迟、显存占用与并发公平性，用数据支持结论。

## 设计原则

- 先建立简单、可信的 eager 基线，再优化；每次优化都保留可比较的参照。
- 先定义状态、不变量和失败边界，再选择类与目录；架构服务于数据流，而不是服务于文件数量。
- 将数值正确性、机制验证和性能收益分开验收。数值一致不能证明 KV 被复用，吞吐变化也不能单独证明原因。
- 一次只引入一个主要变量，记录环境、输入分布、预热方式和重复次数。
- 只在阶段验收完成后打 tag；不复制 `vllm-v0/`、`vllm-v1/` 等整棵目录。

## 计划中的目录

```text
.
├── src/xixii_vllm/       # 唯一实现主线
│   ├── models/           # Qwen3 与权重加载
│   ├── layers/           # RMSNorm、RoPE、Attention、MLP
│   ├── engine/           # 请求、调度器与 Engine
│   ├── cache/            # KV block、页表与前缀复用
│   └── execution/        # batch、runner 与加速路径
├── tests/                # 正确性、机制与 GPU 测试
├── benchmarks/           # 可复现的性能实验
├── docs/                 # 设计说明、ADR 和性能报告
└── examples/             # 最小可运行入口
```

目录会随问题逐步长出来，不在开始时一次性搭出空架构。

## 环境

当前目标环境：Linux x86_64、Python 3.12、NVIDIA RTX 4060 系列 8 GiB、支持 CUDA 13.x
的驱动。默认真实模型为 `Qwen/Qwen3-0.6B`，以给实现、KV Cache 和 profiler 留出显存空间。

先安装 [uv](https://docs.astral.sh/uv/)，然后同步基础开发环境：

```bash
uv sync --locked
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

进入 v3 性能优化阶段后，再安装 GPU 性能依赖：

```bash
uv sync --locked --group perf
```

`perf` 组沿用参考项目验证过的 PyTorch 2.13/CUDA 13.0、FlashAttention 2.8.3 和
FlashInfer 0.6.15 组合。它依赖 Linux x86_64 和兼容的 NVIDIA 驱动，预编译 wheel
也会占用较多下载与磁盘空间。

无需手动激活虚拟环境；命令统一写成 `uv run ...`。编辑器解释器选择
`.venv/bin/python`。

## 每个功能的学习闭环

1. 画出输入、输出、状态变化与关键 shape，并写下自己的性能预测。
2. 写最小测试或对照实验，明确什么结果算通过。
3. 实现最简单的正确版本，和 PyTorch/Hugging Face 参考实现对齐。
4. 用 profiler 或基准数据定位瓶颈，只选择一个主要变量优化。
5. 同时提交代码、测试、设计说明和实验记录，然后复盘取舍。

常用检查命令会随着测试落地保持稳定：

```bash
uv run ruff check .
uv run pytest
```

GPU、真实权重和慢速 benchmark 应使用显式 pytest marker 或独立命令，不混入快速默认测试。

## 阶段验收与 Git tag

- [ ] `v0`：真实 Qwen3-0.6B 能生成；关键层和 logits 有参考对齐；朴素基线有记录。
- [ ] `v1`：prefill/decode 增量结果正确；block 生命周期与 slot mapping 有机制测试；证明减少了重算。
- [ ] `v2`：多请求可持续进入和退出 batch；调度预算与抢占边界有测试；记录吞吐/延迟曲线。
- [ ] `v3`：至少一条 GPU 优化由 profiler 定位并获得可复现收益；说明收益、代价与失效条件。

每阶段完成后再执行类似命令：

```bash
git tag -a v0 -m "validated minimal inference engine"
```

## 参考边界

可以参考 `dzyy-vllm` 的阶段目标、公开 vLLM 设计资料以及 PyTorch、Transformers、
FlashAttention/FlashInfer 的接口文档。核心实现必须先形成自己的数据流、接口契约和测试，
再用参考项目检查遗漏。面试展示时应能解释为什么这样设计、替代方案是什么、数据如何支持结论。

更具体的 AI 协作方式、验收纪律和硬件事实见 [AGENTS.md](AGENTS.md)。
