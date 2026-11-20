import { useEffect, useState, type ComponentType } from "react";
import { parseRoute } from "@/lib/blog-route";

const loadPage = {
  blog: () => import("@/components/ImmersiveBlog").then(module => module.ImmersiveBlog),
  contributions: () => import("@/components/Contributions").then(module => module.Contributions),
};

type Page = keyof typeof loadPage;
function currentPage(): Page {
  return parseRoute(location.hash).view === "contributions" ? "contributions" : "blog";
}

export default function App() {
  const [page, setPage] = useState(currentPage);
  const [loaded, setLoaded] = useState<{page: Page; Component: ComponentType | null}>();
  useEffect(() => {
    const change = () => setPage(currentPage());
    window.addEventListener("hashchange", change);
    change();
    return () => window.removeEventListener("hashchange", change);
  }, []);
  useEffect(() => {
    let cancelled = false;
    // Reveal the route as soon as its code arrives, including on a cold visit.
    loadPage[page]().then(
      Component => { if (!cancelled) setLoaded({page, Component}); },
      () => { if (!cancelled) setLoaded({page, Component: null}); },
    );
    return () => { cancelled = true; };
  }, [page]);
  if (loaded?.page !== page) return <div className="loading-screen" role="status">正在加载博客…</div>;
  const {Component} = loaded;
  if (Component) return <Component />;
  return <div className="loading-screen" role="status">
    <p>博客加载失败。</p><a href="./directory.html">查看文章目录</a>
  </div>;
}
