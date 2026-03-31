https://zhuanlan.zhihu.com/p/2021873950360180314?share_code=15MmMKUXmSWUk&utm_psn=2022324408916738304

https://jannikreinhard.com/2026/02/22/why-cli-tools-are-beating-mcp-for-ai-agents/

Tool：agent调用的工具

MCP：类似于AI的usec标准，**怎么把工具接入大模型的协议层**

SKILLs：让agent懂思考、会协调，可以跨平台复用

CLI：模型执行层，给出skill，之后agent可以去调用CLI。工具的集成度、完善度比mcp、skills的丰富。

MCP是解决的不同llm应用如何调用tools的一个协议；SKILLs是解决如果让agent更好的调用对应tools（tools➕上下文➕领域知识➕简单的调用策略）；CLI解决的是agent如何智能调用一个含丰富tools的集成箱。



为什么有CLI：

1. MCP很花token
2. llm训练中有很多终端交互
3. CLI组合性高

结构化、细权限、非CLI、`tool/list`动态发现tools仍然适用于MCP。



构建CLI优先agent的开发策略

1. 默认优先使用CLI，只有在需要MCP的特定保障时再回退到MCP
2. 为复杂多步骤操作构建包装脚本
3. 使用结构化输出标志符
4. 把`--help`当作动态文档使用



**关键要点**

为可组合性设计（Unix哲学）