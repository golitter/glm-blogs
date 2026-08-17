[Maximizing the value of your Claude Code sessions | Claude by Anthropic](https://claude.com/blog/maximizing-the-value-of-your-claude-code-sessions)

不要把 Agent 会话当作免费的无限记忆；要把它当作一个每轮都会重新参与计算、会积累噪声、但可以通过缓存和隔离进行工程化管理的工作集。



**开始任务**：

1. 新任务开新会话
2. 先确定模型和 effort
3. 只启用需要的 MCP/插件
4. 直接附上已知相关文件
5. 给出目标、约束和验收标准

**任务进行中**：

6. 同一目标继续原会话
7. 命令输出尽量过滤和摘要
8. 高噪声调研交给 Subagent
9. 走错路优先 rewind，而不是不断追加纠正
10. 上下文膨胀时带重点 compact

**结束或离开**：

11. 任务完成后不要让旧会话承载新任务
12. 长时间离开且需要继续：趁缓存热时 compact
13. 重要状态写进代码、文档或交接说明，不只存在聊天中