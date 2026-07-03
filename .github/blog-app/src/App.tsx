import { ArrowUp, ChevronRight, Search } from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { blogTree, markdownCount, recentFiles, updateTime, type BlogTreeNode } from "@/generated/blog-data";
import { cn } from "@/lib/utils";

const REPO_URL = "https://github.com/golitter/glm-blogs";

function CategoryNode({ node, openPaths, togglePath }: {
  node: BlogTreeNode;
  openPaths: Set<string>;
  togglePath: (path: string) => void;
}) {
  const isOpen = openPaths.has(node.path);

  return (
    <div className="rounded-md">
      <button
        type="button"
        className={cn(
          "flex min-h-9 w-full items-baseline justify-between gap-2 rounded-md px-2.5 py-2 text-left transition-colors hover:bg-slate-100 hover:text-sky-800",
          isOpen && "bg-slate-100 text-sky-800",
        )}
        onClick={() => togglePath(node.path)}
      >
        <span className="inline-flex min-w-0 items-center gap-2">
          <ChevronRight
            className={cn("h-3.5 w-3.5 shrink-0 text-slate-500 transition-transform", isOpen && "rotate-90")}
            aria-hidden="true"
          />
          <span className="truncate text-sm font-bold">{node.name}</span>
        </span>
        <span className="shrink-0 text-xs font-semibold text-slate-500">{node.count}</span>
      </button>

      {isOpen ? (
        <div className="ml-4 grid gap-0.5 border-l border-slate-200 pl-3">
          {node.children.map((child) => (
            <CategoryNode key={child.path} node={child} openPaths={openPaths} togglePath={togglePath} />
          ))}
          {node.files.map((file) => (
            <a
              key={file.path}
              href={file.url}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-md px-2.5 py-2 text-sm font-semibold leading-snug text-slate-500 transition-colors hover:bg-slate-100 hover:text-sky-800"
            >
              {file.title}
            </a>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function App() {
  const [query, setQuery] = useState("");
  const [openPaths, setOpenPaths] = useState<Set<string>>(new Set());
  const [showBackToTop, setShowBackToTop] = useState(false);

  const latestFiles = useMemo(() => recentFiles.slice(0, 5), []);

  useEffect(() => {
    const syncBackToTop = () => setShowBackToTop(window.scrollY > 360);
    syncBackToTop();
    window.addEventListener("scroll", syncBackToTop, { passive: true });
    return () => window.removeEventListener("scroll", syncBackToTop);
  }, []);

  useEffect(() => {
    const handleKeydown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "k") {
        event.preventDefault();
        document.getElementById("searchInput")?.focus();
      }
    };
    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, []);

  function performSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    const repoQuery = `repo:golitter/glm-blogs ${trimmed}`;
    window.open(`https://github.com/search?q=${encodeURIComponent(repoQuery)}&type=code`, "_blank");
  }

  function togglePath(path: string) {
    setOpenPaths((current) => {
      const next = new Set(current);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }

  function collapseCategories() {
    setOpenPaths(new Set());
    document.getElementById("categoryPanel")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="sticky top-0 z-10 border-b border-slate-200/80 bg-white/90 backdrop-blur">
        <nav className="mx-auto flex min-h-16 w-full max-w-6xl items-center justify-between gap-6 px-6">
          <div className="flex min-w-0 flex-wrap items-center gap-3">
            <a className="inline-flex items-center gap-2.5 text-base font-bold no-underline" href="./">
              <img
                className="h-7 w-7 rounded-full border border-slate-200 bg-slate-100 object-cover"
                src="./blog-icon.png"
                alt=""
              />
              <span>Golemon Blogs</span>
            </a>
            <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-slate-100 px-2 py-1 text-xs leading-none text-slate-500">
              博客文章 <strong className="text-[13px] font-extrabold text-sky-800">{markdownCount}</strong>
            </span>
          </div>

          <div className="flex items-center gap-5 text-sm text-slate-500">
            <a className="py-2 no-underline hover:text-sky-800" href={REPO_URL} target="_blank" rel="noreferrer">
              Repository
            </a>
            <a
              className="py-2 no-underline hover:text-sky-800"
              href={`${REPO_URL}/commits/master/`}
              target="_blank"
              rel="noreferrer"
            >
              Commits
            </a>
            <a
              className="py-2 no-underline hover:text-sky-800"
              href={`${REPO_URL}/issues`}
              target="_blank"
              rel="noreferrer"
            >
              Issues
            </a>
          </div>
        </nav>
      </header>

      <div className="mx-auto grid min-h-[calc(100vh-4rem)] w-full max-w-[1400px] grid-cols-[200px_minmax(0,960px)_200px] items-start gap-5 px-6 py-8 max-[1120px]:grid-cols-1 max-[1120px]:pt-7">
        <Card className="mt-[92px] max-[1120px]:order-2 max-[1120px]:mt-0">
          <CardHeader className="pb-2">
            <CardTitle>About</CardTitle>
          </CardHeader>
          <CardContent>
            <a className="break-all text-sm leading-7 text-sky-800 no-underline hover:underline" href="https://github.com/golitter" target="_blank" rel="noreferrer">
              https://github.com/golitter
            </a>
          </CardContent>
        </Card>

        <main className="w-full max-[1120px]:order-1">
          <section className="mb-6 grid gap-5">
            <form onSubmit={performSearch} className="rounded-lg border border-slate-200 bg-white p-2 shadow-[0_12px_30px_rgba(26,44,69,0.06)]">
              <div className="relative flex items-center">
                <input
                  id="searchInput"
                  className="h-[52px] w-full rounded-md border border-transparent bg-slate-100 px-4 pr-[58px] text-base text-slate-900 outline-none transition focus:border-sky-300 focus:bg-white focus:ring-4 focus:ring-sky-200/60"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="搜索 Markdown，例如 qwen、agent"
                  autoComplete="off"
                  required
                />
                <Button className="absolute right-1.5 top-1/2 h-10 w-10 -translate-y-1/2 rounded-md p-0" aria-label="搜索">
                  <Search className="h-5 w-5" />
                </Button>
              </div>
            </form>
          </section>

          <div className="grid grid-cols-[220px_minmax(0,1fr)] items-start gap-5 max-[1120px]:grid-cols-1">
            <Card id="categoryPanel">
              <CardHeader className="flex flex-row items-center justify-between gap-3 pb-4">
                <CardTitle>
                  <a className="no-underline hover:text-sky-800" href={REPO_URL} target="_blank" rel="noreferrer">
                    目录
                  </a>
                </CardTitle>
                <Button variant="outline" size="sm" onClick={collapseCategories}>
                  全部折叠
                </Button>
              </CardHeader>
              <CardContent className="grid gap-1">
                {blogTree.map((node) => (
                  <CategoryNode key={node.path} node={node} openPaths={openPaths} togglePath={togglePath} />
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>最近更新</CardTitle>
              </CardHeader>
              <CardContent>
                {latestFiles.map((file) => (
                  <a
                    key={`${file.path}-${file.date}`}
                    className="block rounded-md px-2 py-4 no-underline transition hover:bg-slate-100 [&+&]:border-t [&+&]:border-slate-200"
                    href={file.url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <span className="block text-[15px] font-bold leading-snug text-slate-900">{file.title}</span>
                    <span className="mt-1 block break-all font-mono text-xs leading-snug text-slate-500">{file.path}</span>
                    <span className="mt-1.5 block text-xs font-bold text-amber-700">{file.date}</span>
                  </a>
                ))}
              </CardContent>
            </Card>
          </div>

          <div className="mt-4 text-xs text-slate-500">更新时间: {updateTime} (中国时间)</div>
        </main>

        <Card className="mt-[92px] max-[1120px]:order-3 max-[1120px]:mt-0">
          <CardHeader className="pb-2">
            <CardTitle>搜索提示</CardTitle>
          </CardHeader>
          <CardContent className="text-sm leading-7 text-slate-500">
            <p>如果关键词中间有空格，请使用英文双引号；多个关键词可以用逗号隔开。</p>
            <div className="mt-2 flex flex-wrap gap-2">
              <p>
                例如：<code className="rounded border border-slate-200 bg-slate-100 px-1.5 py-0.5 font-mono text-xs text-slate-900">"go context"</code>
              </p>
              <p>
                例如：<code className="rounded border border-slate-200 bg-slate-100 px-1.5 py-0.5 font-mono text-xs text-slate-900">agent, rag</code>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Button
        className={cn(
          "fixed bottom-6 right-6 z-20 h-10 w-10 rounded-full p-0 shadow-[0_12px_28px_rgba(26,44,69,0.18)] transition-all max-[560px]:bottom-4 max-[560px]:right-4 max-[560px]:h-9 max-[560px]:w-9",
          showBackToTop ? "translate-y-0 opacity-100" : "pointer-events-none translate-y-2 opacity-0",
        )}
        aria-label="回到顶部"
        onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
      >
        <ArrowUp className="h-5 w-5" />
      </Button>
    </div>
  );
}

export default App;
