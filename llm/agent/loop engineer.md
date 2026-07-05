https://arxiv.org/abs/2607.00038

https://mp.weixin.qq.com/s/hcgKahtQRE2QqI6xplv2Rg

https://www.datawhale.cn/activity/483/learn/220/5204?from=homepage

https://agentskillsdev.com/courses/harness-engineering-consensus

https://mp.weixin.qq.com/s/3Zbx4RHB4fOdomI5aA_wIQ

agent这个概念**没有**一个明确的定义。

openai对其的定义为：

```text
Agent = LLM + Memory + 主动规划 + 工具使用
```

之后harness兴起，又成了：
```text
Agent = Harness + Model
```

harness是用工程系统来缩小“大模型不确定性输出”和“实际业务的可交付目标“之间的gap。

>harness 包含：上下文、工具、循环、记忆、状态、权限、钩子、会话、验证、编排和Runtime。



现在又兴起了 loop engineer。

loop更像是一个**为具体任务设计的闭环式**agentic workflow。这里的loop需要结合具体的任务，不再是通用场景。

> workflow：类似于if-else逻辑的链条，整个跳转逻辑是固定的。
>
> agentic wrokflow：workflow加上LLM/Agent node。



> Loop包含自动触发、隔离执行、技能规则、外部连接、多agent分工和持久状态。它让agent从“一次性执行”变成“可持续推进任务的闭环”。



再回顾前面的 Prompt Engineer、Context Engineer。

Prompt Engineer：在turn级或局部任务级优化指令，让 LLM 更好地执行当前请求。。

Context Engineer：在session级或任务环境级维护上下文，让LLM基于符合任务本质、准确相关且低歧义的信息持续推理。



![四个不同engineer范式例子](./loop%20engineer.assets/2f81e68d-63dd-4267-8168-35399c00550d.png)

