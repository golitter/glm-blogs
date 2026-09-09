import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";

import { SearchNotebook } from "./SearchNotebook";
import { blogTree, type BlogFile, type BlogTreeNode } from "@/generated/blog-data";
import { files, parseRoute, routeHash, readLocal, writeLocal, searchFiles, suggestions, highlightedParts, warmSearch, type Quality, type Route } from "@/lib/blog-experience";
import type { createImmersiveBlog } from "@/lib/immersive-blog-scene";

function Highlight({text, query}: {text: string; query: string}) {
  const parts = useMemo(() => highlightedParts(text, query), [text, query]);
  return <>{parts.map((part,i) => part.hit ? <mark key={i}>{part.text}</mark> : part.text)}</>;
}
// The dialog stays mounted while hidden, and hover labels / idle fades re-render the shell on every
// object crossing — memo keeps those renders from reconciling the full results list each time.
const ResultList = memo(function ResultList({results, query, selected, select, remember}: {
  results: BlogFile[]; query: string; selected: number; select: (index: number) => void; remember: (value: string) => void;
}) {
  return <ul id="search-results" className="search-results">{results.map((file,i) => <li id={`search-result-${i}`} key={file.path} className={selected === i ? "selected" : ""}><a href={file.url} target="_blank" rel="noreferrer" onFocus={() => select(i)} onClick={() => remember(query)}><Highlight text={file.title} query={query}/><small><Highlight text={file.path} query={query}/></small></a></li>)}</ul>;
});
function StaticNode({node}: {node: BlogTreeNode}) {
  return <details><summary>{node.name} <small>{node.count} 篇</small></summary><ul>{node.files.map(file => <li key={file.path}><a href={file.url} target="_blank" rel="noreferrer">{file.title}</a></li>)}</ul>{node.children.map(child => <StaticNode key={child.path} node={child}/>)}</details>;
}
// Start fetching the three.js scene while this module evaluates, before React mounts; reuse across quality rebuilds.
const sceneModule: Promise<typeof import("@/lib/immersive-blog-scene")> = import("@/lib/immersive-blog-scene");
// Workflow builds stamp China time at deploy; local dev falls back to the current Shanghai time.
const BUILD_TIME = import.meta.env.VITE_BUILD_TIME
  ?? new Date().toLocaleString("zh-CN", { timeZone: "Asia/Shanghai", hour12: false }).replace(/\//g, "-");
export function ImmersiveBlog() {
  const host = useRef<HTMLDivElement>(null), search = useRef<HTMLInputElement>(null);
  const api = useRef<ReturnType<typeof createImmersiveBlog> | null>(null), applying = useRef(false);
  const [ready, setReady] = useState(false), [error, setError] = useState(false), [simple, setSimple] = useState(false);
  const [editing, setEditing] = useState(false), [query, setQuery] = useState(() => parseRoute(location.hash).q ?? ""), [selected, setSelected] = useState(-1);
  const [label, setLabel] = useState(""), [route, setRoute] = useState(() => parseRoute(location.hash));
  const [tour, setTour] = useState(() => !readLocal("golemon-intro-done", false));
  const introCompleted = useRef(!tour);
  const [awake, setAwake] = useState(true), [settings, setSettings] = useState(false);
  const [quality, setQuality] = useState<Quality>(() => {
    const saved = readLocal<string>("golemon-quality", "");
    return ["high", "balanced", "eco"].includes(saved) ? saved as Quality : matchMedia("(max-width:700px)").matches ? "balanced" : "high";
  });
  const [history, setHistory] = useState<string[]>(() => {
    const saved = readLocal<unknown>("golemon-search-history", []);
    return Array.isArray(saved) ? saved.filter((s): s is string => typeof s === "string").slice(0,6) : [];
  });
  const [random, setRandom] = useState<BlogFile | null>(null);
  const [searchReady, setSearchReady] = useState(false);
  const results = useMemo(() => searchFiles(query), [query, searchReady]);
  function completeIntro() { if (introCompleted.current) return; introCompleted.current = true; setTour(false); writeLocal("golemon-intro-done", true); }
  // useCallback keeps the identity stable across hover-label re-renders so ResultList's memo holds.
  const remember = useCallback(function remember(value: string) {
    if (!value.trim()) return;
    const next = [value.trim(), ...history.filter(s => s !== value.trim())].slice(0,6);
    setHistory(next); writeLocal("golemon-search-history", next);
  }, [history]);
  function navigate(next: Route) { completeIntro(); const hash = routeHash(next); if (location.hash !== hash) location.hash = hash; else api.current?.navigate(next); }
  function navigate3D(next: Route) { setSimple(false); setSettings(false); setRandom(null); navigate(next); }
  function openSearch() { completeIntro(); warmSearch().then(() => setSearchReady(true)); flushSync(() => { setSimple(false); setSettings(false); setEditing(true); setSelected(-1); }); search.current?.focus(); search.current?.select(); }
  function closeSearch() { setEditing(false); host.current?.querySelector("canvas")?.focus(); }
  function submit() { remember(query); navigate({view:"search", q:query}); closeSearch(); }
  function explore() {
    const file = files[Math.floor(Math.random() * files.length)]; if (!file) return;
    // Move to the shelf before presenting a separate, explicit reading choice.
    navigate3D({view:"shelf"});
    setRandom(file);
  }
  useEffect(() => {
    function change() { const next = parseRoute(location.hash); setRoute(next); if (next.view === "search") setQuery(next.q ?? ""); applying.current = true; api.current?.navigate(next); applying.current = false; }
    window.addEventListener("hashchange", change); return () => window.removeEventListener("hashchange", change);
  }, []);
  useEffect(() => {
    // Warm the pinyin search index only after the scene is ready (or in simple mode), so its
    // download + index build never competes with three.js scene initialisation for the main thread.
    if (!ready && !simple) return;
    warmSearch().then(() => setSearchReady(true)).catch(() => {});
  }, [ready, simple]);
  useEffect(() => {
    // A 3D search panel opened before warming finished shows plain-substring fallback results
    // (no pinyin/initials matches); rebuild it once the full index is available.
    if (searchReady && !simple && route.view === "search") api.current?.setQuery(route.q ?? "");
  }, [searchReady]);
  useEffect(() => {
    if (simple || !host.current) return;
    const element = host.current; let cancelled = false; setReady(false); setError(false);
    sceneModule.then(({createImmersiveBlog}) => {
      if (cancelled) return;
      api.current = createImmersiveBlog(element, {
        quality, onReady: () => setReady(true), onSearchRequest: openSearch, onInteraction: completeIntro, onHover: setLabel,
        onError: () => { setError(true); setSimple(true); },
        onRouteChange: (next, replace = false) => {
          if (applying.current || !api.current) return;
          const hash = routeHash(next); setRoute(next);
          if (replace) window.history.replaceState(null, "", hash); else if (location.hash !== hash) location.hash = hash;
        },
      });
      applying.current = true; api.current.navigate(parseRoute(location.hash)); applying.current = false;
    }).catch(() => { if (!cancelled) { setError(true); setSimple(true); } });
    return () => { cancelled = true; api.current?.dispose(); api.current = null; };
  }, [quality, simple]);
  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout>;
    function wake() { setAwake(true); clearTimeout(timeout); timeout = setTimeout(() => setAwake(false), 4500); }
    function shortcut(event: KeyboardEvent) { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") { event.preventDefault(); openSearch(); } }
    wake(); window.addEventListener("pointermove", wake); window.addEventListener("pointerdown", wake); window.addEventListener("keydown", shortcut);
    return () => { clearTimeout(timeout); window.removeEventListener("pointermove", wake); window.removeEventListener("pointerdown", wake); window.removeEventListener("keydown", shortcut); };
  }, []);
  useEffect(() => { setSelected(-1); }, [query]);
  useEffect(() => { document.getElementById(`search-result-${selected}`)?.scrollIntoView({block:"nearest"}); }, [selected]);
  useEffect(() => { if (random && ready && !simple && route.view === "shelf") api.current?.previewArticle(random); }, [random, ready, simple, route.view]);
  return <main className="immersive-root">
    <div ref={host} className="webgl-world" aria-label="Golemon Blogs 三维博客" hidden={simple}/>
    {!ready && !simple && <div className="loading-screen" role="status"><span className="loading-bird">♧</span><p>正在进入三维博客页面…</p><button onClick={() => setSimple(true)}>先看简洁博客</button></div>}
    <nav className={`floating-nav ${awake || editing || settings || simple ? "" : "is-idle"}`} aria-label="快捷导航">
      <button aria-current={route.view === "home" ? "page" : undefined} onClick={() => navigate3D({view:"home"})}>全景</button><button aria-current={route.view === "shelf" || route.view === "category" ? "page" : undefined} onClick={() => navigate3D({view:"shelf"})}>知识目录</button><button aria-current={route.view === "recent" ? "page" : undefined} onClick={() => navigate3D({view:"recent"})}>最近更新</button><button onClick={openSearch}>搜索 <kbd>⌘/Ctrl K</kbd></button><button aria-expanded={settings} onClick={() => {setSimple(false); setSettings(!settings); completeIntro();}}>画质</button><button onClick={explore}>随便看看</button><button onClick={() => {setSimple(!simple); completeIntro();}}>{simple ? "三维博客" : "简洁博客"}</button>
    </nav>

    {settings && <section className="quality-menu" aria-label="画质设置"><label>画质模式<select value={quality} onChange={e => {const value = e.target.value as Quality; setQuality(value); writeLocal("golemon-quality", value);}}><option value="high">高画质 · 完整光影</option><option value="balanced">平衡 · 流畅优先</option><option value="eco">节能 · 低功耗</option></select></label><p>后台暂停绘制，静止时自动降低帧率。</p><button onClick={() => setSettings(false)}>完成</button></section>}
    {tour && ready && !simple && <aside className="first-visit" aria-label="首次访问提示"><strong>欢迎来到 Golemon 的三维博客</strong><p>拖动旋转场景 · 滚轮缩放<br/>点击书本查看分类 · 点击文章前往 GitHub</p><button onClick={completeIntro}>知道了</button></aside>}
    {label && !simple && <div className="object-label" role="status">{label}</div>}
    {simple && <section className="static-directory"><h1>Golemon Blogs · 简洁博客</h1><p className="directory-intro">记录大模型、智能体与系统工程相关的学习与实践。</p>{error && <p role="status">三维渲染暂不可用，所有文章仍可在这里访问。</p>}<p>共 {files.length} 篇文章 · 点击后前往 GitHub · <a href="./directory.html">无脚本目录</a></p>{route.view === "search" ? <><h2>搜索：{route.q || "全部文章"}</h2><button onClick={() => navigate({view:"shelf"})}>返回全部分类</button><ul>{searchFiles(route.q ?? "").map(file => <li key={file.path}><a href={file.url} target="_blank" rel="noreferrer">{file.title}</a></li>)}</ul></> : blogTree.map(node => <StaticNode key={node.path} node={node}/>)}</section>}
    <div className="search-editor" hidden={!editing}><div className="search-notebook">{editing && !simple && <SearchNotebook/>}<form role="dialog" aria-modal="true" aria-labelledby="search-heading" onSubmit={event => {event.preventDefault(); if (selected >= 0 && results[selected]) {remember(query); window.open(results[selected].url,"_blank","noopener,noreferrer");} else submit();}} onKeyDown={event => {
      event.stopPropagation(); if (event.nativeEvent.isComposing) return;
      if (event.key === "Escape") closeSearch();
      if (["ArrowDown","ArrowUp"].includes(event.key)) {event.preventDefault(); setSelected(i => Math.max(0, Math.min(results.length-1, i + (event.key === "ArrowDown" ? 1 : -1))));}
      if (event.key === "Tab") {const fields = Array.from(event.currentTarget.querySelectorAll<HTMLElement>("input,button,a")); const next = (fields.indexOf(document.activeElement as HTMLElement) + (event.shiftKey ? -1 : 1) + fields.length) % fields.length; event.preventDefault(); fields[next]?.focus();}
    }}><h2 id="search-heading">搜索笔记</h2><label htmlFor="blog-search">标题、路径、拼音或首字母</label><input ref={search} id="blog-search" type="search" value={query} onChange={e => setQuery(e.target.value)} autoComplete="off" aria-label="搜索博客文章" aria-controls="search-results" aria-activedescendant={selected >= 0 ? `search-result-${selected}` : undefined} onKeyDown={e => {if (e.key === "Enter" && e.nativeEvent.isComposing) e.preventDefault();}}/><p aria-live="polite">匹配 {results.length} 篇 · ↑↓ 选择，回车打开</p>
    {!query && history.length > 0 && <div className="search-history"><span>最近搜索</span>{history.map(term => <button type="button" key={term} onClick={() => setQuery(term)}>{term}</button>)}<button type="button" onClick={() => {setHistory([]); writeLocal("golemon-search-history", []);}}>清空</button></div>}
    <ResultList results={results} query={query} selected={selected} select={setSelected} remember={remember}/>
    {!results.length && <div><p>没有找到文章，试试这些相近分类：</p>{suggestions(query).map(node => <button key={node.path} type="button" onClick={() => {navigate({view:"category",path:node.path}); closeSearch();}}>{node.path}</button>)}</div>}
    <div className="search-actions"><button type="button" onClick={closeSearch}>取消</button><button type="button" onClick={submit}>在书页中查看</button></div></form></div></div>
    {random && <aside className="random-card" role="dialog" aria-label="随机探索"><small>今天翻到这一篇</small><h2>{random.title}</h2><p>{random.path}</p><a href={random.url} target="_blank" rel="noreferrer" onClick={() => setRandom(null)}>前往 GitHub 阅读 ↗</a><button onClick={explore}>再抽一篇</button><button onClick={() => setRandom(null)}>收起</button></aside>}
    <a className="accessible-directory" href="./directory.html">完整文章目录（无需三维渲染）</a>
    <div className={`build-stamp ${awake ? "" : "is-idle"}`}><span>最后更新 {BUILD_TIME}（中国时间）</span></div>
  </main>;
}
