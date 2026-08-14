> 使用aicoding对开源aicoding进行分析。

## `/goal` 功能

> 来源于codex

持久化目标 → turn 内工作 → turn 结束记账 → idle 检测决定是否继续 → 循环直到终止。

> 在持久化目标中，可能会产生一个类似todos的plan。借助hooks机制实现。
>
> > hooks 是 core 提供的"挂载点"，goal 扩展是"挂钩子的人"，循环逻辑写在 goal 里，靠 core 触发 hooks 来驱动。这种"core 提供插座、扩展提供电器"的设计，让 `/goal` 成了一个可插拔的独立功能。



## todolist

> codex、claudecode

对于todolist的实现，claudecode结合了harness，而codex则没有结合codex。 

todolist机制是：agent的系统提示词里面有todolist的对应提示词。由agent判断什么时候需要todolist，之后会调用todolist工具产生todolist plan。之后agent不断的turn去完成这个todolist内容。一般是每完成一个todo就勾选一个，直到彻底完成。 



## 推理强度

> codex、及[Codex、Claude Code的推理档位，其实就是一句提示词。](https://mp.weixin.qq.com/s/Rf32OCWDM-mf1ocfKKQbUw)

reasoning effort 是 API 层的 `reasoning.effort` 参数，模型侧据此实际分配推理算力；客户端不靠改提示词来"模拟"更高强度。

> 推理强度是 harness 通过 reasoning.effort 参数传给模型的信号；这个信号能生效，是因为模型在训练时被配了与各强度对应的长度惩罚，从而学会了"高强度→写更长草稿→输出更多 token、消耗更多推理算力"。



## AGENTS.md 导入

> codex

AGENTS.md 在 agent 每 turn 调用前导入:用户级(~/.codex/AGENTS.md)+ 项目级(根→cwd 目录链各层 AGENTS.md)拼接注入上下文。项目级共享 32 KiB 预算、先到先得,超限即截断或丢弃。



## 上下文

```text
┌─────────────────────────────────────────────────────────┐
│ 顶层 instructions 字段                                   │  ← base_instructions
│ = gpt_5_2_prompt.md 渲染后的系统提示词                    │     (Responses API 的 instructions)
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ [developer 消息] 初始上下文包(可选,全量注入时)          │
│  ├── 开发者指令(developer_instructions)                 │
│  ├── 模型切换指令                                        │
│  ├── token 预算提示(TokenBudgetContext)                 │
│  └── 扩展贡献的 developer 片段                           │
├─────────────────────────────────────────────────────────┤
│ [user 消息] 上下文用户包                                 │
│  ├── # AGENTS.md instructions ... </INSTRUCTIONS>       │  ← AGENTS.md(用户级+项目级)
│  ├── 推荐插件提示                                        │
│  ├── 环境信息(cwd/日期/时区/网络/文件系统)              │  ← environments section
│  ├── 权限策略(沙箱/审批/执行策略)                       │  ← permissions section
│  ├── 人格状态、协作模式、context window 指导 等          │
│  └── 多 agent 模式提示(如适用)                          │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ [user] 用户第1条输入                                     │  ← 真实对话开始
│ [assistant] 模型回复 + Reasoning(思考)                  │
│ [tool_call] FunctionCall(如 apply_patch)                 │
│ [tool_result] FunctionCallOutput(工具结果,已截断)        │
│ [user] 用户第2条输入                                     │
│ ...                                                      │  ← ContextManager.items
│ [user] 当前轮用户输入 / 工具结果                          │
└─────────────────────────────────────────────────────────┘

```

关键点:world state section(AGENTS.md、环境、权限等)不是放在 system prompt 里,而是作为 user/developer 角色的独立消息插在历史开头和后续增量更新里。后续轮次中只有变化的 section 会以新消息追加(**带 replace 提示**)。

world state section有近二十种类，分别分到了developer、user提示词。

| Section ID | 注入条件           | 用途                                                         |
| :--------- | :----------------- | :----------------------------------------------------------- |
| 1          | 总是               | 当前模型 slug；若与上次不同，发出模型切换指令                |
| 2          | 配置了人格         | 人格变体（friendly / pragmatic / none）                      |
| 3          | TokenBudget 特性   | 上下文窗口预算指导文本                                       |
| 4          | 实时模式           | 实时会话的开始/结束指令                                      |
| 5          | 有 AGENTS.md       | AGENTS.md 指令（用户级+项目级）                              |
| 6          | 总是               | 沙箱模式、审批策略、执行策略                                 |
| 7          | 有已批准命令       | 已保存的「免审批命令前缀」                                   |
| 8          | 总是               | 协作模式（默认 / plan 模式等）                               |
| 9          | 总是               | 核心环境信息：cwd、日期、时区、网络策略、文件系统沙箱、可用子 agent |
| 10         | 有延迟执行器       | 延迟执行器（远程环境）的使用说明                             |
| 11         | Apps 特性开启      | 应用/连接器的使用提示                                        |
| 12         | Plugins 特性开启   | 插件使用提示                                                 |
| 13         | 有延迟工具命名空间 | 延迟加载的工具命名空间说明                                   |
| 14+        | 扩展贡献           | 由扩展（plugins/MCP）动态注册的 section                      |
| 15         | 多 agent           | 根 agent / 子 agent 的使用提示                               |
| 16         | 多 agent 模式      | 多 agent 模式状态                                            |

> > 对话历史放最后**是 cache 设计的需要**,append-only 的历史让前面整段稳定前缀每轮都命中缓存,所以中等长度对话越长反而越省。真正的代价出现在**历史长到触发压缩**时——摘要重写历史会破坏前缀,导致缓存大面积失效,这是 Codex 用「压缩后重建初始上下文 + 截断工具输出 + 90% 阈值」等多项设计去缓解的核心矛盾。

压缩只对对话历史进行:按字节数÷4 估算 token 触发,保留最近的用户消息(≤20K token),对更早的对话调用 LLM 生成摘要,最后用「最近用户消息 + 摘要」填充替代旧历史。初始上下文包不压缩,重新注入。

```text
[初始上下文包]              ← 不压缩,重新注入(instructions + AGENTS.md + 环境 + 权限)
[摘要]                      ← 前缀 + LLM 对旧对话的总结
[最近用户消息 ≤20K token]    ← 原文保留(中间的 assistant/工具调用已并入摘要)
[当前轮新内容]              ← 继续对话

```

Codex 没有从技术上解决多次压缩导致的信息逐级丢失,而是:用 window_number 追踪深度、把窗口状态告诉模型让其自适应、保留最近用户消息和初始上下文延缓衰减,并在每次压缩后明确警告用户「多次压缩会降低准确性,建议开新线程」。本质上把质量保证的责任交给了用户(及时开新线程),而非自动保持长线程的准确度。



## 会话标题

> codex cli、codex桌面端、claudecode IDE插件

codex cli是再会话首次用户消息时，对消息进行过滤得到的标题。

codex桌面端和claudecode IDE插件大概是首次用户消息的前几次turn中，调用类似ai摘要的方式来产生的。



## plugin

Plugin 是一个可分发的打包单元，通过marketplace/git/npm/remote 安装到本地，用一个 plugin.json 清单声明它携带的能力。加载时，它把这些能力注册进 Codex 运行时的各个子系统：skills（进 skill 发现管线）、MCP servers（进 MCP catalog）、apps/connectors、hooks。模型本身不"调用 plugin"，而是使用它注册出来的那些底层能力。