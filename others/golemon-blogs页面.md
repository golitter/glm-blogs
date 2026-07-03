# Golemon Blogs 页面本地运行

页面源码在 `.github/blog-app`，技术栈是 React 19、Vite 8、TypeScript、Tailwind CSS 和 shadcn/ui 风格组件。页面数据不是写死的，运行前会扫描仓库里的 Markdown 文件，生成目录树、最近更新和文章数量。

## 安装依赖

在仓库根目录执行：

```bash
cd .github/blog-app
pnpm install
```

## 开发预览

```bash
cd .github/blog-app
python3 scripts/generate-blog-data.py
pnpm dev
```

浏览器打开终端里显示的本地地址，一般是：

```text
http://127.0.0.1:5173/
```

如果新增、删除或移动了 Markdown 文件，需要重新执行：

```bash
python3 scripts/generate-blog-data.py
```

然后刷新页面。

## 构建静态页面

```bash
cd .github/blog-app
python3 scripts/generate-blog-data.py
./node_modules/.bin/tsc --noEmit
./node_modules/.bin/vite build
```

构建结果会生成到：

```text
.github/blog-app/dist
```

## 在仓库根目录查看构建结果

如果想像之前一样打开根目录的 `dist/index.html`，可以把构建结果同步过去：

```bash
cp -R .github/blog-app/dist/. dist/
```

然后打开：

```text
dist/index.html
```

## GitHub Pages 构建

`.github/workflows/blog-page.yaml` 会在 GitHub Actions 中自动执行：

```bash
pnpm install --frozen-lockfile
python3 scripts/generate-blog-data.py
./node_modules/.bin/tsc --noEmit
./node_modules/.bin/vite build
```

最终发布 `.github/blog-app/dist`。
