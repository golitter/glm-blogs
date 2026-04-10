#!/usr/bin/env python3
"""Course CLI parser - 解析 /course 命令参数。

用法:
    python parse.py "/course plan"
    python parse.py "/course content --stage 3 --no-review"
    python parse.py "/course summary 重点标注事务部分"
    python parse.py "/course content 不要面试资料 --stage 2"
    python parse.py "/course status"

输出 JSON:
    {
        "command": "plan|content|summary|status|auto",
        "flags": {"stage": null, "no_review": false},
        "extra": "附加指令文本"
    }
"""

import json
import re
import sys


def parse(input_str: str) -> dict:
    """解析 /course 命令字符串。"""
    text = input_str.strip()

    # 去掉 /course 前缀
    if text.startswith("/course"):
        text = text[len("/course"):].strip()
    else:
        return {"error": "不是 /course 命令"}

    if not text:
        return {
            "command": "auto",
            "flags": {"stage": None, "no_review": False},
            "extra": "",
        }

    # 已知的 commands
    valid_commands = {"plan", "content", "summary", "status", "help"}

    # 提取 command
    tokens = text.split()
    command = tokens[0] if tokens[0] in valid_commands else "auto"
    rest = text[len(tokens[0]):].strip() if command != "auto" else text

    # 提取 flags
    flags = {"stage": None, "no_review": False}

    # --stage <n>
    stage_match = re.search(r"--stage\s+(\d+)", rest)
    if stage_match:
        flags["stage"] = int(stage_match.group(1))
        rest = rest[:stage_match.start()] + rest[stage_match.end():]

    # --no-review
    if "--no-review" in rest:
        flags["no_review"] = True
        rest = rest.replace("--no-review", "")

    # 清理多余空白得到 extra
    extra = " ".join(rest.split())

    return {
        "command": command,
        "flags": flags,
        "extra": extra,
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python parse.py '<course 命令字符串>'")
        sys.exit(1)

    input_str = " ".join(sys.argv[1:])
    result = parse(input_str)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
