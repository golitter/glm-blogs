Skills 扩展了 Claude 能做的事情。创建一个 `SKILL.md` 文件，其中包含说明，Claude 会将其添加到其工具包中。Claude 在相关时使用 skills，或者你可以使用 `/skill-name` 直接调用一个。

- 触发方式
  - 自动：Claude 根据 `description` 判断“当前对话跟这个 skill 相关”，就自动加载
  - 手动：你输入 `/skill-name` 直接调用



典型结构：

```shell
my-skill/
├── SKILL.md           # 主说明（必需）
├── template.md        # 模板（可选）
├── examples/
│   └── sample.md      # 示例输出（可选）
└── scripts/
    └── validate.sh    # Claude 可以执行的脚本（可选）

```



例如，[everything-claude-code/skills/pytorch-patterns at main · affaan-m/everything-claude-code (github.com)](https://github.com/affaan-m/everything-claude-code/tree/main/skills/pytorch-patterns)

在项目的`.claude/skills`下面创建`python-pattern`目录，

里面添加`SKILL.md`，将md文档内容复制到里面即可进行调用。

```shell
/python-pattern
```



例子2：

描述：A test skill to verify skills are working. When user types "xxx", respond with "yyy" and run the time script.

创建`xxx`目录，

**SKILL.md**:

```markdown
---
name: xxx
description: A test skill to verify skills are working. When user types "xxx", respond with "yyy" and run the time script.
---

# Test Skill

This is a test skill to verify the skills system is working correctly.

## Trigger

When the user sends the message **"xxx"** (exactly, case-sensitive), you MUST:

1. Output: `yyy`
2. Run the Python script at `./claude/skills/xxx/scripts/get_time.py`

## Rules

1. If user message is exactly "xxx":
   - First, output `yyy`
   - Then, execute the script using Bash tool: `python ./claude/skills/xxx/scripts/get_time.py`
2. Show the script output to the user

## Example

User: xxx
Assistant:
yyy

[RUNNING SCRIPT]
Current time: 2026-03-22 15:15:30
Timestamp: 1742672130.123

```

`skills/xxx/scripts/get_time.py`：

```python
#!/usr/bin/env python3
"""Script to get current time for testing skills."""

from datetime import datetime

def main():
    now = datetime.now()
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timestamp: {now.timestamp()}")

if __name__ == "__main__":
    main()

```

使用

```shell
/xxx
```

或者输入`xxx`。