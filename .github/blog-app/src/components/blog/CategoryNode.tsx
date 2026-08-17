import { ChevronRight } from "lucide-react";

import type { BlogTreeNode } from "@/generated/blog-data";
import type { CategoryTone } from "@/lib/category-style";
import { cn } from "@/lib/utils";

export function CategoryNode({
  node,
  openPaths,
  togglePath,
  tone,
  depth = 0,
}: {
  node: BlogTreeNode;
  openPaths: Set<string>;
  togglePath: (path: string) => void;
  tone?: CategoryTone;
  depth?: number;
}) {
  const isOpen = openPaths.has(node.path);
  const isTopLevel = depth === 0;

  return (
    <div>
      <button
        type="button"
        className={cn(
          "grid min-h-10 w-full grid-cols-[auto_minmax(0,1fr)] items-center gap-2 border-2 border-transparent px-2.5 py-2 text-left transition-[background-color,transform,box-shadow] hover:border-ink hover:bg-sun hover:shadow-[3px_3px_0_#111] active:translate-x-[3px] active:translate-y-[3px] active:shadow-none",
          isOpen && (isTopLevel ? "border-ink bg-sun shadow-[3px_3px_0_#111]" : "border-ink bg-surface"),
        )}
        style={
          isOpen && !isTopLevel && tone
            ? {
                backgroundColor: tone.bg,
              }
            : undefined
        }
        onClick={() => togglePath(node.path)}
      >
        <span className="inline-flex items-center gap-2 text-ink">
          <ChevronRight
            className={cn(
              "h-4 w-4 shrink-0 stroke-[3] text-ink transition-transform duration-150",
              isOpen && "rotate-90",
            )}
            aria-hidden="true"
          />
          {tone ? (
            <span
              aria-hidden
              className="h-3 w-3 shrink-0 border-2 border-ink"
              style={{ backgroundColor: tone.ink }}
            />
          ) : null}
        </span>
        <span className="inline-flex min-w-0 items-baseline gap-2">
          <span
            className={cn(
              "truncate text-sm font-bold text-ink-soft",
              isOpen ? "font-black text-ink" : "font-bold",
            )}
          >
            {node.name}
          </span>
          <span className="shrink-0 border border-ink bg-surface px-1.5 font-mono text-[10px] font-black tabular-nums text-ink">
            {node.count}
          </span>
        </span>
      </button>

      {isOpen ? (
        <div className="ml-[17px] mt-2 grid min-w-0 gap-1 overflow-hidden border-l-[3px] border-line pl-3">
          {node.children.map((child) => (
            <CategoryNode
              key={child.path}
              node={child}
              openPaths={openPaths}
              togglePath={togglePath}
              tone={tone}
              depth={depth + 1}
            />
          ))}
          {node.files.map((file) => (
            <a
              key={file.path}
              href={file.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group/file grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1.5 overflow-hidden border-2 border-transparent px-2.5 py-2 text-[13px] font-semibold leading-snug text-ink-muted no-underline transition-colors hover:border-ink hover:bg-primary hover:text-white"
            >
              <span className="min-w-0 truncate">{file.title}</span>
              <span aria-hidden className="shrink-0 opacity-0 transition-opacity group-hover/file:opacity-100">
                ↗
              </span>
            </a>
          ))}
        </div>
      ) : null}
    </div>
  );
}
