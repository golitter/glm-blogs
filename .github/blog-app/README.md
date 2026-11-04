# Golemon Blogs Page

Golemon Blogs 的前端站点。扫描仓库里的 Markdown，生成一个**目录树 + 最近更新 + 搜索**的导航页面，部署在 GitHub Pages。

🔗 线上：<https://golitter.github.io/glm-blogs/>

## 技术栈

React 19 · Vite 8 · TypeScript · Tailwind CSS v4 · Three.js · pnpm

## 快速开始

> 所有 `pnpm` 命令必须在本目录（`.github/blog-app`）下执行。

```bash
pnpm install
pnpm dev            # 开发预览（注意：不会自动重新生成数据）
```

新增 / 删除 / 移动 Markdown 后，`pnpm dev` 不会刷新数据，需手动重跑：

```bash
node scripts/generate-data.mjs
```

## 构建

```bash
pnpm build          # = prebuild(自动跑 python 生成数据) + tsc --noEmit + vite build
```

产物在 `dist/`。运行 `pnpm preview`，通过终端显示的本地 HTTP 地址预览，不能直接双击 HTML 文件。生成数据的启动脚本自动选择 Python 命令并开启 UTF-8，兼容 Windows 与 Linux。

## 目录结构

```
scripts/generate-blog-data.py   扫描 *.md → 生成数据（可见性规则在这里配），并产出无脚本目录 public/directory.html
src/
  App.tsx                        入口，渲染 ImmersiveBlog
  main.tsx                       挂载入口
  index.css                      设计 token 与全站样式
  generated/blog-data.ts         ⛔ 自动生成，勿手改
  components/ImmersiveBlog.tsx   三维博客的 React 壳（导航 / 搜索 / 画质 / 简洁降级）
  components/SearchNotebook.tsx  搜索窗口的 WebGL 封皮
  lib/immersive-blog-scene.ts    Three.js 主场景（书房 / 书架 / 更新板 / 书页面板）
  lib/blog-experience.ts         搜索（拼音/首字母）、hash 路由、NEW 判定
  lib/shelf-layout.ts            书架排布纯函数
  lib/constants.ts               外部链接
vite.config.ts                   base: "./"（相对路径，兼容 GitHub Pages 子路径）
```

架构、关键文件与常见任务的完整对照见 [AGENTS.md](AGENTS.md)。

## 内容从哪来

页面内容**不是写死的**，由 [`scripts/generate-blog-data.py`](scripts/generate-blog-data.py) 扫描整个仓库的 Markdown 生成。

**哪些 md 会出现在前端**由脚本里的可见性规则决定（自动排除 + 黑名单 `EXCLUDED_PATHS` + 白名单 `INCLUDE_ONLY`）。完整说明见 [`../../others/golemon-blogs页面.md`](../../others/golemon-blogs页面.md) 的「数据流与内容可见性」一节。

## 部署

[`../workflows/blog-page.yaml`](../workflows/blog-page.yaml) 在 GitHub Actions 上执行 `pnpm install --frozen-lockfile` → 生成数据 → `tsc --noEmit` → `vite build`，发布 `dist/` 到 GitHub Pages。

## 更多文档

- [页面设计与组件 / 内容可见性](../../others/golemon-blogs页面.md)
- [AGENTS.md](AGENTS.md) — 给 AI agent 的操作手册

## 开源贡献

顶部「开源贡献」入口和 `#/contributions` 展示公开且已合并的 PR。页面按项目分组，项目按 GitHub Stars 降序排列，每个项目内按合并时间倒序排列。顶部总数来自实际展示的已合并 PR；项目及 PR 均链接至 GitHub，支持展开、收起和移动端阅读。

数据快照是 `src/generated/contributions.json`，由脚本生成，不要手动修改。在本目录执行：

```bash
pnpm sync:contributions
```

脚本从 `src/lib/constants.ts` 中的 GitHub 主页读取账号，分页查找公开的已合并 PR，再逐条核验合并状态并获取各项目的 Stars。可使用本机已有的 GitHub CLI 登录，或通过环境变量 `GH_TOKEN` / `GITHUB_TOKEN` 提供凭据；凭据不会写入快照或发送至浏览器。全部校验成功后才替换快照，同步失败保留原有数据。

采集和恢复已发布数据共用严格校验：账号、合并标记、时间、PR 编号、GitHub 链接及项目关联必须匹配，重复记录和未来的同步日期会被拒绝。分页期间结果总数变化也会中止本次更新，保留原快照供下次重试。当前单次查询最多支持 1,000 条 PR，超过时明确报错，不会静默截断；达到这个规模时需要按合并日期拆分查询。

本地开发及普通构建直接读取已保存的快照，不需要访问 GitHub。页面默认展开所有项目，以浅色标题条区分项目，不显示日期。

GitHub Actions 已配置每五个北京时间自然日更新一次 PR 和 Stars：每天北京时间 09:17 检查上次成功发布的数据，未满五天时跳过同步、构建和发布。跨月及跨年仍保持五天间隔，失败后下一天自动重试。也可以在 Actions 页面手动运行此工作流，立即强制同步。

构建会生成公开的数据快照 `public/contributions.json` 并随页面发布。后续定时检查及普通博客发布会读取这份已发布快照，避免恢复为仓库中较旧的数据。自动更新不创建 Git 提交；PR 检查只构建测试，不同步或发布。

定时任务需要这些改动进入默认分支后才会生效。GitHub 的定时触发可能延后；公开仓库连续 60 天无活动时，定时任务会被平台自动停用，需要重新启用，参见 [GitHub 定时任务说明](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)。

## 全屏 3D 云端博客

主场景采用全屏 WebGL Canvas，搜索时按需创建独立的三维手账。南小鸟灵感的奶油白、浅绿、灰棕和香槟金贯穿实体书架、更新留言板、搜索设备、个人相框与三维博客。

- `src/components/ImmersiveBlog.tsx`：挂载三维世界；点击搜索时打开可见的原生输入窗口，支持中文输入法、回车提交、取消与 Esc 关闭。
- `src/lib/immersive-blog-scene.ts`：创建完整 Three.js 场景、镜头导航、射线交互、分类书本、更新卡片、搜索结果和滚动立体手账。
- 场景、目录与书页结果使用 WebGL。另有轻量 HTML 快捷导航、首次引导、搜索窗口、画质设置与简洁博客，保证触摸、键盘、中文输入法和辅助阅读可用。
- 分类和文章继续读取 `src/generated/blog-data.ts`，点击文章在新窗口打开真实 GitHub 链接。
- 支持拖动旋转、滚轮缩放、物体悬停描边、镜头飞行、Bloom、实体阴影及移动端较低渲染分辨率。

检查页面时应覆盖书本点击、手账滚动、最近更新链接、搜索中文输入、Esc 返回、镜头导航和手机视口。

镜头根据视口比例计算取景距离，阅读时锁定旋转；仅保留放大的顶部快捷导航，移除场景内四块重复导航牌和底部导航。全景使用居中的完整书房取景；知识目录聚焦书架，最近更新聚焦更新板，所选入口高亮。仅全景的偏心镜头链接恢复时自动校正，局部视角保留自己的镜头中心。书架每排五本，超过 10 个分类后自动向上增加第三排、第四排及后续层板；柜体、侧板、标题和镜头取景随高度适配，分类全部同时显示，不分页。分类书页保留子目录层级，返回按钮和 Esc 逐级返回。文章目录和搜索结果使用固定大小的三维书页，内部连续滚动，不再按十篇分页。支持滚轮、拖动、方向键、PageUp/PageDown、Home/End；边界裁切同时约束点击区域，阅读时滚轮不再缩放镜头。搜索支持标题与路径匹配和空结果提示。文章正文继续通过真实 GitHub 链接打开。

分类容量回归检查：Node 24 下运行 `node --test scripts/shelf-layout.test.mjs`，覆盖 0、1、10、11、15、16、20、21、103 个分类，检查排数、全部分类可见、书本不重叠以及柜体边界。

场景以奶油色拱窗、绿植、吊灯、金色灯串、地毯和带页边的实体书本组成。WebGL 创建失败或上下文丢失时自动显示简洁博客，可切回三维重试。开发期间修改动态加载的场景模块后，需刷新预览以重新创建场景。



场景支持水平 360° 环绕和俯仰观察，可查看侧面及背面。背面补有墙面木条、铭牌、柜体固定条、更新板背撑和补光。点击全景恢复正面取景；打开文章书页后拖动仍用于内部滚动。

## 交互与性能

- 首次操作后隐藏引导，localStorage 记录完成状态；快捷导航闲置 4.5 秒淡出，移动、触摸或键盘聚焦恢复。
- 搜索支持标题、路径、大小写、拼音、首字母与多个关键词，实时统计、高亮、最近六次查询、方向键选择和回车打开；Ctrl/Command+K 打开搜索。拼音字典增加了首屏脚本体积，新增依赖为 pinyin-pro。
- Hash 地址支持 `#/category/llm/agent`、`#/recent`、`#/search?q=codex`；镜头位置写入 camera 查询参数且使用 replaceState，不会为每次拖动堆叠历史。分类跳转使用正常历史，支持刷新和后退。
- 高画质保留阴影、描边、Bloom 和最高 1.7 倍像素比；平衡模式关闭阴影与 Bloom，像素比最高 1.25；节能模式直接渲染、降低圆角与球体分段、关闭抗锯齿，像素比最高 1。手机默认平衡。模式保存在本地，切换时释放旧场景并从 URL 恢复。
- document.hidden 时取消渲染调度；静止后降低至约 6–12 fps，交互时恢复。具体设备帧率和功耗仍受硬件影响。
- 简洁博客是同一数据的可访问索引，分类和子目录默认折叠，点击展开；构建时另生成 `public/directory.html`，无脚本也能访问全部 GitHub 文章。
- Git 最新文件变更提供时间和新增／修改状态；NEW 只标记最近七天新增文章。书本厚度随数量增长，近期更新更明亮，超过 90 天稍褪色，子目录数量增加书脊装饰。
- 随便看看会抽取真实文章、聚焦并抽出其所属分类书本；只有点击确认链接才前往 GitHub。

运行 `node --test scripts/experience.test.mjs scripts/shelf-layout.test.mjs` 检查搜索、拼音高亮、URL 编解码、NEW 期限、无脚本索引与书架增长。生成数据的 Git 时间来自提交记录，未跟踪文章不虚构更新时间。

搜索窗口使用 WebGL 封皮、纸页、金属装订环与书签，随指针轻微倾转；原生输入和结果列表叠放在纸面以支持中文输入及键盘。模型按需绘制，关闭搜索时释放资源；简洁模式保留无 WebGL 的输入表单。
