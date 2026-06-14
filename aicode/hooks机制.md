# Claude Code / Codex / OpenCode Hook 机制

> 参考文档：[Claude Code Hooks](https://docs.anthropic.com/en/docs/claude-code/hooks) · [Codex Hooks](https://developers.openai.com/codex/hooks) · [OpenCode 插件](https://opencode.ai/docs/zh-cn/plugins/) · [OpenCode 插件系统详解](https://zhuanlan.zhihu.com/p/2027144829352583703)

Hook = 在生命周期节点自动执行自定义逻辑。

> Hook = 事件 + 匹配规则 + 处理器

三个 AI 编程工具都内置了 hook。差异主要在事件名格式和通信方式，本质一致。下面分别看。

## Claude Code / Codex：命令行 + stdin JSON

配置里按事件名注册一个 command，事件触发时把 JSON 通过 stdin 喂给脚本，脚本读完 `exit 0`。

```jsonc
// Claude Code: ~/.claude/settings.json   —— 事件名 PascalCase
{ "hooks": { "PreToolUse": [{ "hooks": [{ "command": "pet-hook.sh claude-code", "type": "command" }] }] } }

// Codex: ~/.codex/hooks.json              —— 事件名 snake_case
{ "hooks": { "pre_tool_use": [{ "hooks": [{ "type": "command", "command": "pet-hook.sh codex" }] }] } }
```

差异：

- 事件名：Claude `PascalCase`，Codex `snake_case`
- 字段名：Claude 固定 `hook_event_name` + `session_id`；Codex 更杂，session id 可能是 `session_id` / `sessionId` / `conversation_id` / `thread_id`
- 信任门：Codex 独有，改了 hook 命令要在 `/hooks` 按 hash 手动 Trust 一次

## OpenCode：进程内 TS 插件

不是独立进程，是 OpenCode 启动时 import 运行的 TS 模块，导出 handler 映射。事件名 `dot.case`，通信走函数参数 `(input, output)`：

```typescript
export const PetPlugin = async ({ directory }) => ({
  "tool.execute.before": async (input) => { /* input.tool, input.sessionID */ },
  event: async ({ event }) => { /* event.type, event.properties.sessionID */ },
});
```

两个坑：session id 是大写 D 的 `sessionID`；事件型的 session id 在 `event.properties.sessionID` 里，不是 `event.sessionID`。

## 三平台对比

| | Claude Code | Codex | OpenCode |
|---|---|---|---|
| 机制 | 命令行 + stdin JSON | 命令行 + stdin JSON | 进程内 JS/TS 插件 |
| 事件名 | `PascalCase` | `snake_case` | `dot.case` |
| 注册事件数 | 11 | 9 | 8（6 event + 2 tool 拦截） |
| session id | `session_id` | `session_id` / `sessionId` / `conversation_id` / `thread_id` | `sessionID`（大写 D） |
| 信任 | 改了在 `/hooks` 审查 | 需 review & trust（按 hash） | 无 |
| 阻断 | `exit 2` | `decision: "block"` | `throw Error` |

事件名查表前先归一化（Codex `snake_case`、OpenCode `dot.case` → `PascalCase`），之后即可统一处理。
