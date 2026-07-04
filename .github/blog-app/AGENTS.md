# AGENTS.md

给在本项目（`.github/blog-app`）工作的 AI agent 的上下文。

## 这是什么

Golemon Blogs 的前端：一个 React + Vite 单页应用，扫描整个仓库的 Markdown，渲染成**目录树 + 最近更新 + 搜索**的导航站，部署到 GitHub Pages。**页面内容来自生成的数据，不是写死的。**

## 红线（动手前必读）

1. **不要手改 `src/generated/blog-data.ts`**。它是 `scripts/generate-blog-data.py` 生成的。要改内容 / 结构，改 python 脚本，然后重新生成。
2. **所有 `pnpm` 命令在 `.github/blog-app` 下执行**。仓库根目录没有 `package.json`，在根跑会 `ERR_PNPM_NO_PKG_MANIFEST`。
3. **`pnpm dev` 不会重新生成数据**。增删改 md 或改了可见性规则后，手动跑 `python3 scripts/generate-blog-data.py` 再刷新。`pnpm build` 的 `prebuild` 会自动跑。
4. **`pnpm build` 会跑 `tsc --noEmit`**，类型错误和未使用的导入会被拦下。改完用它验证。

## 数据管线

```
仓库 *.md  →  scripts/generate-blog-data.py  →  src/generated/blog-data.ts  →  React 组件
```

脚本做两件事：

- `build_tree()`：`rglob("*.md")` 按可见性规则过滤，构建目录树 + 文章数
- `build_recent_files()`：`git log` 取最近改动的前 5 篇

导出：`blogTree` / `recentFiles` / `markdownCount` / `updateTime`。

## 内容可见性规则

在 `scripts/generate-blog-data.py` 的 `is_ignored_rel_path()` + 顶部两个常量：

- **自动排除**：`dist`、任一路径段以 `.` 开头（`.github` / `.claude` / ...）、文件名为 `readme.md`
- **`EXCLUDED_PATHS`**（黑名单）：隐藏整个目录或单个文件
- **`INCLUDE_ONLY`**（白名单）：某目录下只展示列出的文件

优先级：自动 ≈ 黑名单 > 白名单。详见 [`../../others/golemon-blogs页面.md`](../../others/golemon-blogs页面.md) 的「数据流与内容可见性」一节。

> 关键特性：目录树是从 md 路径反推的——一个目录下若没有任何够格的 md，该目录不会出现。

## 关键文件

| 文件 | 职责 |
|---|---|
| `scripts/generate-blog-data.py` | 扫描 md、生成数据、可见性规则（改内容逻辑改这里） |
| `src/generated/blog-data.ts` | 生成产物，勿手改 |
| `src/App.tsx` | 三栏布局骨架 |
| `src/index.css` | `@theme` 设计 token（配色 / 字体 / 阴影），改 token 全站联动 |
| `src/components/blog/` | CategoryTree / CategoryNode / SearchBar / RecentUpdates |
| `src/components/layout/` | SiteHeader / SiteFooter |
| `src/components/sidebar/` | AboutCard / SearchTips |
| `src/lib/` | constants（链接）/ category-style（分类配色）/ utils（cn） |
| `vite.config.ts` | `base: "./"`（相对路径，GitHub Pages 子路径兼容） |

## 常见任务

| 任务 | 怎么做 |
|---|---|
| 隐藏某个目录 / 文件 | 改 `EXCLUDED_PATHS`，重跑 python |
| 某目录只展示指定 md | 改 `INCLUDE_ONLY`，重跑 python |
| 改配色 / 字体 / 氛围 | 改 `src/index.css` 的 `@theme` token |
| 改「最近更新」条数 | 同时改 `RecentUpdates.tsx` 的 `slice(0, N)` **和** python 的 `len(rows) >= N` |
| 加导航 / 账号链接 | `layout/SiteHeader.tsx` 或 `sidebar/AboutCard.tsx` + `lib/constants.ts` |

## 改动检查清单

- [ ] 改了 md / 可见性规则 → `python3 scripts/generate-blog-data.py`
- [ ] 改了代码 → `pnpm build`（tsc + vite 都过）
- [ ] 生成产物 `blog-data.ts` 已重新生成，不手改

## 更多文档

- [页面设计与组件 / 内容可见性](../../others/golemon-blogs页面.md)
