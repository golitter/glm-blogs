from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
OUTPUT_PATH = APP_ROOT / "src" / "generated" / "blog-data.ts"
REPO_URL = "https://github.com/golitter/glm-blogs"
BRANCH = "master"
TZ = timezone(timedelta(hours=8))

# ---- 前端可见性规则 ----
# 路径均为相对仓库根目录，用 "/" 分隔。两套互补规则：
#
# 1) EXCLUDED_PATHS（黑名单）：隐藏整个目录 或 某个具体文件
#    "some/dir"          -> 该目录下所有 md（任意层级）都不展示
#    "path/to/file.md"   -> 仅隐藏这一个文件
#
# 2) INCLUDE_ONLY（按目录白名单）：某目录下「只展示」列出的文件，其余全部隐藏
#    "skills": {"skills/index.md"}  -> skills 下只展示 index.md
#    键是目录路径，值是该目录下允许展示的文件全路径集合
#
# 优先级：黑名单 > 白名单（被 EXCLUDED_PATHS 命中的一定隐藏，即使在白名单里也一样）
EXCLUDED_PATHS: set[str] = {
    # "some/dir",
    # "path/to/file.md",
}

INCLUDE_ONLY: dict[str, set[str]] = {
    "skills": {"skills/index.md"},
}


def is_ignored_rel_path(rel_path: str) -> bool:
    """Decide whether a markdown path is hidden from the blog frontend.

    Three layers, checked in order of precedence:
      1. Build output / dot-prefixed paths (.git, .github, .claude, .DS_Store...)
      2. EXCLUDED_PATHS — hide a whole directory (prefix match) or a single file
      3. INCLUDE_ONLY — if the file sits under a restricted directory, it is
         shown only when explicitly listed in that directory's allowlist
    """
    parts = rel_path.split("/")
    if "dist" in parts or any(part.startswith(".") for part in parts):
        return True
    if any(rel_path == p or rel_path.startswith(p + "/") for p in EXCLUDED_PATHS):
        return True
    for dir_key, allowed in INCLUDE_ONLY.items():
        if rel_path.startswith(dir_key + "/") and rel_path not in allowed:
            return True
    return False


def is_markdown_file(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return (
        path.suffix == ".md"
        and path.name.lower() != "readme.md"
        and not is_ignored_rel_path(rel)
    )


def github_url(kind: str, rel_path: str) -> str:
    encoded = "/".join(quote(part) for part in rel_path.split("/"))
    return f"{REPO_URL}/{kind}/{BRANCH}/{encoded}"


def new_node(name: str, path: str) -> dict:
    return {"name": name, "path": path, "children": {}, "files": []}


def add_file(tree: dict, rel_path: str) -> None:
    parts = rel_path.split("/")
    current = tree
    current_path: list[str] = []
    for part in parts[:-1]:
        current_path.append(part)
        path = "/".join(current_path)
        current = current["children"].setdefault(part, new_node(part, path))

    filename = parts[-1]
    title = filename[:-3] if filename.endswith(".md") else filename
    current["files"].append(
        {
            "title": title,
            "path": rel_path,
            "url": github_url("blob", rel_path),
        }
    )


def finalize_node(node: dict) -> dict:
    children = [finalize_node(child) for _, child in sorted(node["children"].items(), key=lambda item: item[0].lower())]
    files = sorted(node["files"], key=lambda item: item["title"].lower())
    count = len(files) + sum(child["count"] for child in children)
    return {
        "name": node["name"],
        "path": node["path"],
        "count": count,
        "children": children,
        "files": files,
    }


def build_tree() -> tuple[list[dict], int]:
    root = new_node("", "")
    for md_path in sorted(REPO_ROOT.rglob("*.md")):
        if is_markdown_file(md_path):
            add_file(root, md_path.relative_to(REPO_ROOT).as_posix())

    children = [finalize_node(child) for _, child in sorted(root["children"].items(), key=lambda item: item[0].lower())]
    root_files = sorted(root["files"], key=lambda item: item["title"].lower())
    if root_files:
        children.append(
            {
                "name": "root",
                "path": "root",
                "count": len(root_files),
                "children": [],
                "files": root_files,
            }
        )

    return children, sum(child["count"] for child in children)


def build_recent_files() -> list[dict]:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "log", "--name-only", "--pretty=format:%ct|%H", "-40", "--", "*.md"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    current_time = ""
    seen: set[str] = set()
    rows: list[tuple[int, str]] = []

    for line in result.stdout.splitlines():
        if "|" in line and line.split("|", 1)[0].isdigit():
            current_time = line.split("|", 1)[0]
            continue

        rel_path = line.strip()
        if (
            current_time
            and rel_path.endswith(".md")
            and Path(rel_path).name.lower() != "readme.md"
            and not is_ignored_rel_path(rel_path)
            and rel_path not in seen
            and (REPO_ROOT / rel_path).is_file()
        ):
            seen.add(rel_path)
            rows.append((int(current_time), rel_path))
            if len(rows) >= 5:
                break

    rows.sort(reverse=True, key=lambda item: item[0])
    recent = []
    for timestamp, rel_path in rows:
        filename = Path(rel_path).name
        title = filename[:-3] if filename.endswith(".md") else filename
        recent.append(
            {
                "title": title,
                "path": rel_path,
                "url": github_url("blob", rel_path),
                "date": datetime.fromtimestamp(timestamp, TZ).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return recent


def ts_const(name: str, value: object, satisfies: str | None = None) -> str:
    suffix = f" satisfies {satisfies}" if satisfies else ""
    return f"export const {name} = {json.dumps(value, ensure_ascii=False, indent=2)}{suffix};"


def main() -> None:
    blog_tree, markdown_count = build_tree()
    recent_files = build_recent_files()
    update_time = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

    OUTPUT_PATH.write_text(
        "\n".join(
            [
                "export type BlogFile = {",
                "  title: string;",
                "  path: string;",
                "  url: string;",
                "};",
                "",
                "export type BlogTreeNode = {",
                "  name: string;",
                "  path: string;",
                "  count: number;",
                "  children: BlogTreeNode[];",
                "  files: BlogFile[];",
                "};",
                "",
                "export type RecentFile = BlogFile & {",
                "  date: string;",
                "};",
                "",
                ts_const("blogTree", blog_tree, "BlogTreeNode[]"),
                ts_const("recentFiles", recent_files, "RecentFile[]"),
                ts_const("markdownCount", markdown_count),
                ts_const("updateTime", update_time),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Generated {OUTPUT_PATH.relative_to(REPO_ROOT)} with {markdown_count} markdown files")


if __name__ == "__main__":
    main()
