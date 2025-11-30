[[2412.15115\] Qwen2.5 Technical Report (arxiv.org)](https://arxiv.org/abs/2412.15115)

[【LLM技术报告】Qwen2.5技术报告（全文） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/13936916587)



qwen2.5稠密模型0.5b到72b，api服务的MoE模型turbo、plus模型。

对于稠密模型架构和tokenizer采样qwen2的并进行优化：

- 分组查询注意力
- 旋转位置编码
- QKV偏置
- RMSNorm

在稠密模型的基础上，MoE模型是将标准的前馈网络FFN层替换为MoE层，**每个层包含多个FFN专家，并通过路由机制将token分配给top-k专家**。

tokenizer采用的是qwen的tokenizer，实现了字节级别的字节对编码（BBPE）。