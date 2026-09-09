import { useEffect, useRef } from "react";

/** A real WebGL notebook surrounds native, accessible editing controls. */
export function SearchNotebook() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let cancelled = false;
    let dispose: (() => void) | undefined;
    import("@/lib/search-notebook-scene").then(({createSearchNotebook}) => {
      if (!cancelled && ref.current) dispose = createSearchNotebook(ref.current);
    }).catch(() => { /* The paper-colored native form remains usable without WebGL. */ });
    return () => { cancelled = true; dispose?.(); };
  }, []);
  return <div ref={ref} className="search-notebook-model" aria-hidden="true"/>;
}
