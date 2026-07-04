# Golemon Blogs Page

Golemon Blogs 的前端站点。扫描仓库里的 Markdown，生成一个**目录树 + 最近更新 + 搜索**的导航页面，部署在 GitHub Pages。

🔗 线上：<https://golitter.github.io/glm-blogs/>

## 技术栈

React 19 · Vite 8 · TypeScript · Tailwind CSS v4 · shadcn/ui 风格基础组件 · lucide-react · pnpm

## 快速开始

> 所有 `pnpm` 命令必须在本目录（`.github/blog-app`）下执行。

```bash
pnpm install
pnpm dev            # 开发预览（注意：不会自动重新生成数据）
```

新增 / 删除 / 移动 Markdown 后，`pnpm dev` 不会刷新数据，需手动重跑：

```bash
python3 scripts/generate-blog-data.py
```

## 构建

```bash
pnpm build          # = prebuild(自动跑 python 生成数据) + tsc --noEmit + vite build
```

产物在 `dist/`。在仓库根目录预览：`cp -R dist/. ../../dist/` 后打开 `../../dist/index.html`。

## 目录结构

```
scripts/generate-blog-data.py   扫描 *.md → 生成数据（可见性规则在这里配）
src/
  App.tsx                        三栏布局骨架
  main.tsx                       入口
  index.css                      设计 token（@theme：配色/字体/阴影）
  generated/blog-data.ts         ⛔ 自动生成，勿手改
  components/                     ui / layout / blog / sidebar
  lib/                            constants / category-style / utils
vite.config.ts                   base: "./"（相对路径，兼容 GitHub Pages 子路径）
```

## 内容从哪来

页面内容**不是写死的**，由 [`scripts/generate-blog-data.py`](scripts/generate-blog-data.py) 扫描整个仓库的 Markdown 生成。

**哪些 md 会出现在前端**由脚本里的可见性规则决定（自动排除 + 黑名单 `EXCLUDED_PATHS` + 白名单 `INCLUDE_ONLY`）。完整说明见 [`../../others/golemon-blogs页面.md`](../../others/golemon-blogs页面.md) 的「数据流与内容可见性」一节。

## 部署

[`../workflows/blog-page.yaml`](../workflows/blog-page.yaml) 在 GitHub Actions 上执行 `pnpm install --frozen-lockfile` → 生成数据 → `tsc --noEmit` → `vite build`，发布 `dist/` 到 GitHub Pages。

## 更多文档

- [页面设计与组件 / 内容可见性](../../others/golemon-blogs页面.md)
- [AGENTS.md](AGENTS.md) — 给 AI agent 的操作手册
