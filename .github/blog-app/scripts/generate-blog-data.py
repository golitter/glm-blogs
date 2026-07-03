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


def is_markdown_file(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT)
    parts = rel.parts
    return (
        path.suffix == ".md"
        and path.name.lower() != "readme.md"
        and ".git" not in parts
        and "dist" not in parts
        and not rel.as_posix().startswith(".github/")
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
