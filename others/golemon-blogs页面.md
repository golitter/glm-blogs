# Golemon Blogs 页面

页面源码在 `.github/blog-app`，技术栈：React 19 + Vite 8 + TypeScript + Tailwind CSS v4 + shadcn/ui 风格的基础组件（`ui/card`、`ui/button`），图标用 lucide-react，字体自托管 `@fontsource-variable/geist`。

页面数据不写死：由 `scripts/generate-blog-data.py` 扫描仓库里的 Markdown 文件，生成目录树、最近更新和文章数到 `src/generated/blog-data.ts`。

整体视觉是**编辑式（editorial）· 暖色油墨单色**风格：骨色画布、近黑油墨字、发丝边框、克制用色，追求"笔记库 / 文档站"的质感，而不是常见的 SaaS 蓝紫。

> ⚠️ 所有 `pnpm` 命令都必须在 `.github/blog-app` 目录下执行（只有这里有 `package.json`）；在仓库根目录跑会报 `ERR_PNPM_NO_PKG_MANIFEST`。

## 一、快速运行

### 安装依赖

```bash
cd .github/blog-app
pnpm install
```

### 开发预览

```bash
cd .github/blog-app
python3 scripts/generate-blog-data.py   # 生成数据
pnpm dev                                 # 仅启动 vite，不会自动重新生成数据
```

浏览器打开终端提示的地址，默认 `http://127.0.0.1:5173/`（端口被占用会自动顺延）。

> `pnpm dev` **不会**触发 `prebuild`，所以新增/删除/移动 Markdown 后，要手动重跑一次 `python3 scripts/generate-blog-data.py`，再刷新页面。

### 构建静态页面

```bash
cd .github/blog-app
pnpm build   # = prebuild(自动跑 python) + tsc --noEmit + vite build
```

`build` 会自动先跑 `prebuild` 重新生成数据，产物在 `.github/blog-app/dist`。

### 在仓库根目录查看构建结果

```bash
cp -R .github/blog-app/dist/. dist/
```

然后打开 `dist/index.html`。

### GitHub Pages 自动构建

`.github/workflows/blog-page.yaml` 会在 GitHub Actions 里执行：`pnpm install --frozen-lockfile` → 生成数据 → `tsc --noEmit` → `vite build`，最终发布 `.github/blog-app/dist`。

## 二、页面风格与设计系统

所有设计 token 集中在 [`src/index.css`](.github/blog-app/src/index.css) 的 `@theme` 块。Tailwind v4 会把每个 `--color-*` / `--font-*` / `--shadow-*` 自动注册成工具类（`bg-canvas`、`text-ink`、`border-line`、`font-mono`、`shadow-soft`…），并支持透明度修饰（如 `bg-ink/10`、`ring-ink/15`）。

**核心原则：改 token = 全站联动，尽量不要在组件里写死十六进制色值。**

### 字体

| token | 用途 | 当前值 |
|---|---|---|
| `--font-sans` | 正文 / UI（默认） | Geist Variable，回退 Inter / 系统 / PingFang SC |
| `--font-mono` | 数字、路径、`<kbd>` | Geist Mono，回退系统等宽 |
| `--font-serif` | 衬线（当前未使用，masthead 已移除，保留备用） | 跨平台衬线，含中文宋体回退 |

字体按 `unicode-range` + `font-display: swap` 加载，**运行时只会下载 latin 子集**；中文走系统字体回退，不阻塞首屏。

### 配色（暖色油墨单色）

| token | 含义 | 值 |
|---|---|---|
| `--color-canvas` | 页面画布（骨色） | `#f7f6f2` |
| `--color-surface` | 卡片表面 | `#fdfdfb` |
| `--color-recess` | 凹陷 / 选中态底色 | `#f0eee8` |
| `--color-ink` | 主文字（近黑，非纯黑） | `#1c1b19` |
| `--color-ink-soft` | 次级文字 | `#4b4944` |
| `--color-ink-muted` | 灰文字 | `#8a8780` |
| `--color-ink-faint` | 最淡（序号/占位） | `#b8b5ac` |
| `--color-line` | 发丝边框 | `#e7e5df` |
| `--color-line-strong` | 加重边框 / 焦点 | `#d8d5cc` |
| `--color-tag-*-bg/ink` | 分类配色（sage/clay/amber/plum/slate/sky） | 低饱和淡彩 |

"强调色就是油墨本身"：按钮默认 `bg-ink text-canvas`，链接 hover 变 `ink`，焦点环用 `ring-ink/15`。彩色仅用于目录分类的语义标记（见 `lib/category-style.ts`）。

### 阴影 / 纹理 / 动效

- `--shadow-soft` / `--shadow-lift`：暖色调、单一顶向光源、超弥散阴影。
- `body::before`：固定层的 SVG 噪点（`feTurbulence`，`pointer-events:none`），给画布一点"纸张颗粒"，不随滚动重绘。
- 动画只用 `transform` / `opacity`（GPU 友好）；入场用 CSS 类 `animate-fade-in-up`，配合内联 `animationDelay` 做错落进场；`@media (prefers-reduced-motion)` 下全部关闭。
- `<kbd>` 在 `index.css` 里有统一样式（物理键帽感），搜索框的 `⌘K`、搜索提示里的按键都用它。

### 布局（[`src/App.tsx`](.github/blog-app/src/App.tsx)）

三栏网格 `[224px_1fr_248px]`，外层 `max-w-[1240px]`，视口 ≤1120px 折叠为单列：

- 左 aside：`AboutCard`（身份卡：头像 + GitHub / CSDN 链接）
- 主区：`SearchBar`（命令栏，带 `⌘K`）→ 内层 `[1fr_320px]`：`CategoryTree`（宽、主）+ `RecentUpdates`（窄）
- 右 aside：`SearchTips`
- 顶部 `SiteHeader`（sticky 导航 + 文章数徽标）、底部 `SiteFooter`、右下 `BackToTop`

## 三、组件结构

```
src/
  App.tsx                       布局骨架（三栏 + header/footer）
  components/
    ui/card.tsx, button.tsx     基础原语（按钮变体集中在 buttonVariants）
    layout/SiteHeader.tsx       顶部导航 + 文章数徽标
    layout/SiteFooter.tsx       页脚（说明 + 更新时间）
    blog/SearchBar.tsx          搜索框（⌘K 聚焦，提交跳 GitHub code search）
    blog/CategoryTree.tsx       目录容器 + "全部折叠"
    blog/CategoryNode.tsx       递归树节点（展开/折叠、文件链接、分类色块）
    blog/RecentUpdates.tsx      最近更新列表（序号 + NEW 标）
    sidebar/AboutCard.tsx       左侧身份卡
    sidebar/SearchTips.tsx      右侧搜索语法提示
    BackToTop.tsx               回到顶部
  lib/
    constants.ts                REPO_URL / PROFILE_URL / CSDN_URL / REPO_LINKS
    category-style.ts           分类 pastel 配色映射 tagFor(index)
    utils.ts                    cn() = clsx + tailwind-merge
  generated/blog-data.ts        ⛔ 自动生成，勿手改
```

## 四、常见修改技巧

| 想改什么 | 改哪里 | 怎么改 |
|---|---|---|
| 整站配色 / 氛围 | `src/index.css` 的 `@theme` | 改对应 `--color-*`，全站联动 |
| 字体 | `src/index.css` 的 `--font-*` | 换字体栈；新增自托管字体走 `@fontsource` |
| 顶部导航链接 | `layout/SiteHeader.tsx` + `lib/constants.ts` | 加一个 `<NavLink>`，URL 放进 `REPO_LINKS` |
| 左侧账号链接 | `sidebar/AboutCard.tsx` + `lib/constants.ts` | 现有 GitHub(`PROFILE_URL`)/CSDN(`CSDN_URL`)，照着加一行 `<a>` |
| 目录分类的小色块 | `lib/category-style.ts` 的 `TONES` | 增删/调整 pastel 对，按 index 循环分配 |
| 目录树行 / 缩进 / 引导线 | `blog/CategoryNode.tsx` | 调 `ml-`、`border-l`、chevron、tone chip |
| "最近更新"条数 | `blog/RecentUpdates.tsx` 的 `slice(0, 5)` **和** `scripts/generate-blog-data.py` 的 `len(rows) >= 5` | 两处一起改（python 决定源数据上限） |
| 搜索占位符 / 行为 | `blog/SearchBar.tsx` | `placeholder`；提交逻辑在 `performSearch` |
| 卡片 / 按钮样式 | `ui/card.tsx` / `ui/button.tsx` | 改原语，所有调用处自动跟随 |
| 加新组件 | `src/components/...` | 用 `@/components/...` 别名导入，在 `App.tsx` 挂载 |

改完务必 `cd .github/blog-app && pnpm build` 验证 —— `tsc --noEmit` 会拦住类型错误和未使用的导入。

## 五、数据流与注意事项

- 数据流：`scripts/generate-blog-data.py`（扫 `*.md` + `git log` 取最近更新）→ `src/generated/blog-data.ts`（导出 `blogTree` / `recentFiles` / `markdownCount` / `updateTime`）。
- `src/generated/blog-data.ts` **是生成文件，不要手动编辑**；要改逻辑去改 python 脚本。
- `pnpm build` 的 `prebuild` 会自动重跑 python；`pnpm dev` 不会，需手动跑。
- 所有 `pnpm` 命令在 `.github/blog-app` 下执行。

## 六、性能要点

- 字体：`unicode-range` + `font-display: swap`，运行时只下载 latin 子集，不阻塞首屏。
- 动画：仅 `transform` / `opacity`；图片 `decoding="async"` + 固定宽高，防布局抖动。
- JS 以 React + ReactDOM 为主（gzip 后约 76KB），其余依赖很小；如需让首访首屏更快，可后续上 SSG 预渲染（当前未启用）。
