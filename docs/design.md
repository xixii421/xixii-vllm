# Qwen3 eager 前向基线

## 边界

本阶段实现无 KV Cache 的 Qwen3 dense decoder 前向，作为后续缓存和调度优化的数值基线。
模型规模不写死，由 Hugging Face checkpoint 自带的 `Qwen3Config` 决定；默认
`Qwen/Qwen3-0.6B` 配置实例化后有 596,049,920 个唯一参数。

模块边界如下：

- `model.py` 只拥有模型拓扑：Embedding、Decoder 堆叠、末尾 RMSNorm 和 LM Head。
- `layer/` 按组件组织计算算子：RMSNorm、RoPE、GQA Attention、因果 mask 和 SwiGLU MLP。
- 参数名保持 Hugging Face Qwen3 checkpoint 的命名，以便严格加载并暴露缺失权重。

## 数据流和 shape

```text
input_ids [B, S]
  -> embedding [B, S, D]
  -> N * decoder(hidden states [B, S, D])
       -> Q [B, Hq, S, Dh]
       -> K/V [B, Hkv, S, Dh]
       -> GQA 展开及 causal attention [B, Hq, S, Dh]
       -> output projection + residual [B, S, D]
       -> SwiGLU MLP + residual [B, S, D]
  -> final norm [B, S, D]
  -> lm_head logits [B, S, V]
```

关键不变量：

- `Hq % Hkv == 0`，每个 KV head 服务固定数量的 query heads。
- `Dh` 为偶数，RoPE 才能将最后一维分成两半旋转。
- Q/K 在每个 head 的 `Dh` 维上先做 RMSNorm，再应用 RoPE。
- 第 `t` 个 token 只允许关注 `[0, t]`，padding token 不能作为 key 被访问。
- `tie_word_embeddings=true` 时，Embedding 与 LM Head 必须共享同一块参数存储。

## 当前取舍和验证

当前使用独立 Q/K/V 投影和 eager attention。相比融合 QKV 或 SDPA/FlashAttention，
它更容易逐项检查 shape 和与参考实现对齐；确认性能瓶颈后再替换计算路径。

快速测试以同一组权重对齐 Transformers Qwen3 eager 实现，覆盖 fp32、bf16、padding mask、
因果性和 tied embedding。KV Cache、增量 decode、真实权重端到端生成与 GPU 性能不属于本次
验收范围，不能据此宣称 v0 已完成。
