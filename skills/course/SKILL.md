---
name: course
description: 通用分阶段学习助手。根据用户需求生成学习计划，按阶段给出学习内容，学完后生成重点摘要笔记。用户通过 /course 调用。
---

# Course - 分阶段学习助手

帮助用户按阶段系统学习任意技术主题。

## Usage

```
/course <command> [flags] [extra]
```

- `command`：plan / content / summary / status / help（省略时自动判断下一步）
- `flags`：`--no-review`、`--stage <n>`
- `extra`：命令后的自由文本，作为附加指令

完整文档见 `docs/usage.md`。

## Trigger

仅在用户发送 `/course` 开头的消息时触发。不响应自然语言。

## Workflow

### Step 1：解析命令

用户输入 `/course ...` 后，先运行解析脚本提取 command / flags / extra：

```bash
python3 .claude/skills/course/scripts/parse.py "<用户输入的完整字符串>"
```

脚本输出 JSON：
```json
{
    "command": "plan|content|summary|status|help|auto",
    "flags": {"stage": null|int, "no_review": false|true},
    "extra": "附加指令文本"
}
```

根据解析结果路由：

```
command
  ├── plan     → sub/plan.md
  ├── content  → sub/content.md
  ├── summary  → sub/summary.md
  ├── status   → 输出当前进度
  ├── help     → 读取并展示 docs/usage.md
  └── auto     → 自动判断：
                    无学习计划 → plan
                    有计划无阶段内容 → content
                    有阶段内容 → 提示继续学习或执行 summary
```

### `/course plan [extra]`

1. 询问用户：当前水平、学习目标、可投入时间
2. 根据回答 + extra 附加指令生成 `学习计划.md`
3. 生成完毕后自动执行 content，生成阶段一学习内容

详细规范见 `sub/plan.md`。

### `/course content [extra]`

1. 读取 `学习计划.md`，找到第一个未勾选 `[ ]` 的阶段
2. 读取之前阶段的 `notes/阶段X.md`（如有），了解已学内容
3. 生成 `阶段X.md`，包含：
   - 准备工作（复用之前阶段的资源）
   - 上阶段简要复习（非第一阶段时，除非指定 `--no-review`）
   - 知识点讲解 + 可运行示例
   - 实战练习 + 面试速查表 + 练习建议
4. extra 中的自由文本作为附加要求融入生成内容

详细规范见 `sub/content.md`。

### `/course summary [extra]`

1. 读取当前 `阶段X.md`，结合对话中的提问和易错点
2. 生成 `notes/阶段X.md` 重点摘要（精简、表格为主、不含代码示例）
3. extra 中的自由文本作为附加要求融入生成内容
4. 提示用户可执行 `/course content` 进入下一阶段

详细规范见 `sub/summary.md`。

### `/course status`

读取 `学习计划.md`，输出：
- 总阶段数
- 已完成阶段（`[x]`）
- 当前阶段（第一个 `[ ]`）
- 当前阶段是否已有内容文件

### `/course help`

读取 `docs/usage.md` 并展示给用户。

## Flags

| Flag | 说明 | 适用 command |
|------|------|-------------|
| `--no-review` | 跳过上阶段复习 | content |
| `--stage <n>` | 指定阶段编号 | content, summary |

## Rules

1. 仅在 `/course` 命令时触发，不响应自然语言
2. 所有生成的代码示例必须可直接运行
3. 阶段之间内容递进，不重复已学概念
4. 笔记（notes/）是精简摘要，不含代码和练习，表格为主
5. 每个阶段 md 文件末尾固定结构：面试速查 → 练习建议 → 核心能力总结

## Directory Convention

```
项目根目录/
├── 学习计划.md              ← 学习计划（阶段清单）
├── 阶段一.md                ← 阶段一详细学习内容
├── 阶段二.md                ← 阶段二详细学习内容
├── ...
├── notes/
│   ├── 阶段一.md            ← 阶段一重点摘要
│   ├── 阶段二.md            ← 阶段二重点摘要
│   └── ...
└── .claude/skills/course/
    ├── SKILL.md             ← 主 skill（本文件）
    ├── docs/
    │   └── usage.md         ← CLI 使用手册
    ├── scripts/
    │   └── parse.py         ← CLI 参数解析脚本
    └── sub/
        ├── plan.md          ← 规划流程详细规范
        ├── content.md       ← 生成阶段学习内容
        └── summary.md       ← 生成阶段重点摘要
```
