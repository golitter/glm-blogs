import { useEffect, useRef, useState } from "react";

/** The illustration covers loading, import failures and lost WebGL contexts. */
export function Atelier() {
  const host = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [staticMode, setStaticMode] = useState(false);
  useEffect(() => {
    if (staticMode) return;
    const element = host.current;
    if (!element) return;
    let cancelled = false;
    let dispose: (() => void) | undefined;
    const observer = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      observer.disconnect();
      import("@/lib/atelier-scene").then(({ createAtelier }) => {
        if (cancelled) return;
        try {
          dispose = createAtelier(element, () => setReady(true), () => setReady(false));
        } catch {
          setReady(false);
        }
      }).catch(() => { if (!cancelled) setReady(false); });
    }, { rootMargin: "120px" });
    observer.observe(element);
    return () => { cancelled = true; observer.disconnect(); dispose?.(); };
  }, [staticMode]);
  return <><div className={`atelier ${ready && !staticMode ? "is-ready" : ""}`} role="img" aria-label="可拖动观察的立体云端小屋：浅绿蝴蝶结小鸟、打开的手账、书本和金色星星">
    <img className="atelier-fallback" src="./atelier-fallback.svg" alt="" width="640" height="500" />
    <div ref={host} className="atelier-canvas" aria-hidden="true" />
    <span className="scene-sticker sticker-one" aria-hidden="true">a pocket of sunshine</span>
    <span className="scene-sticker sticker-two" aria-hidden="true">✧</span>
  </div><div className="scene-controls"><span aria-hidden="true">↔</span><span>{staticMode ? "当前为静态画面" : "拖动场景，查看立体空间"}</span><button className="scene-mode" type="button" aria-pressed={staticMode} onClick={() => { setReady(false); setStaticMode(!staticMode); }}>{staticMode ? "开启 3D" : "静态模式"}</button></div></>;
}
