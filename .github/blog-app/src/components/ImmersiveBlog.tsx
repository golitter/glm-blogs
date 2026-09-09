import { useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

export function ImmersiveBlog() {
  const host = useRef<HTMLDivElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const sceneApi = useRef<{ dispose: () => void; setQuery: (value: string) => void } | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(false);
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    const element = host.current;
    if (!element) return;
    let cancelled = false;
    import("@/lib/immersive-blog-scene").then(({ createImmersiveBlog }) => {
      if (cancelled) return;
      sceneApi.current = createImmersiveBlog(element, {
        onReady: () => setReady(true),
        onSearchRequest: () => {
          flushSync(() => setEditing(true));
          search.current?.focus({ preventScroll: true });
          search.current?.select();
        },
      });
    }).catch(() => { if (!cancelled) setError(true); });
    return () => {
      cancelled = true;
      sceneApi.current?.dispose();
      sceneApi.current = null;
    };
  }, []);

  return <main className="immersive-root">
    <div ref={host} className="webgl-world" aria-label="Golemon Blogs 三维博客" />
    <div className={`loading-screen ${ready ? "is-hidden" : ""}`} aria-live="polite">
      <span className="loading-bird">♧</span>
      <p>{error ? "三维场景未能加载，请重试" : "正在打开云端手账…"}</p>
      {error && <button onClick={() => window.location.reload()}>重新加载</button>}
    </div>
    <div className="search-editor" hidden={!editing}>
    <form role="dialog" aria-modal="true" aria-labelledby="search-heading" onSubmit={(event) => {
      event.preventDefault();
      sceneApi.current?.setQuery(search.current?.value ?? "");
      setEditing(false);
      host.current?.querySelector("canvas")?.focus();
    }} onKeyDown={(event) => {
      event.stopPropagation();
      if (event.key === "Escape") { setEditing(false); host.current?.querySelector("canvas")?.focus(); }
      if (event.key === "Tab") {
        const fields = Array.from(event.currentTarget.querySelectorAll<HTMLElement>("input, button"));
        const next = (fields.indexOf(document.activeElement as HTMLElement) + (event.shiftKey ? -1 : 1) + fields.length) % fields.length;
        event.preventDefault(); fields[next]?.focus();
      }
    }}>
    <h2 id="search-heading">搜索笔记</h2>
    <label htmlFor="blog-search">输入标题或目录关键词</label>
    <input
      ref={search}
      id="blog-search"
      type="search"
      aria-label="搜索博客文章"
      placeholder="输入关键词"
      autoComplete="off"
      onKeyDown={(event) => { if (event.key === "Enter" && event.nativeEvent.isComposing) event.preventDefault(); }}
    />
    <div className="search-actions">
      <button type="button" onClick={() => { setEditing(false); host.current?.querySelector("canvas")?.focus(); }}>取消</button>
      <button type="submit">搜索</button>
    </div>
    </form>
    </div>
    <nav className="sr-navigation" aria-label="辅助导航">
      <a href="https://github.com/golitter/glm-blogs">访问博客仓库</a>
      <a href="https://github.com/golitter">访问 Golemon 的 GitHub</a>
    </nav>
  </main>;
}
