[A global workspace in language models \ Anthropic](https://www.anthropic.com/research/global-workspace)

[Jacobian Lens ｜ Neuronpedia](https://www.neuronpedia.org/jlens)

J-lens 捕获模型推理过程中的 residual-stream 中间状态，并通过 Jacobian 将其中具有未来语言表达能力的内部表示解码成人类可读的概念。

J-lens 相当于模型内部的“调试器”，把推理过程中的高维激活转换成人类可读的概念。除了攻击检测，它还能识别欺骗、评测意识和数据造假，追踪推理与幻觉的形成过程，并定位模型答错的具体环节。未来也可用于 Agent 执行高风险操作前的内部审查，以及指导安全训练和模型优化。