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
          "rounded-xl grid min-h-11 w-full grid-cols-[auto_minmax(0,1fr)] items-center gap-2 border border-transparent px-2.5 py-2 text-left transition-[background-color,transform,box-shadow] hover:border-line hover:bg-sun hover:shadow-soft active:scale-[0.98]  active:shadow-none",
          isOpen && (isTopLevel ? "border-line bg-sun shadow-soft" : "border-line bg-surface"),
        )}
        style={
          isOpen && !isTopLevel && tone
            ? {
                backgroundColor: tone.bg,
              }
            : undefined
        }
        aria-expanded={isOpen}
        onClick={() => togglePath(node.path)}
      >
        <span className="inline-flex items-center gap-2 text-ink">
          <ChevronRight
            className={cn(
              "h-4 w-4 shrink-0 stroke-[1.8] text-ink transition-transform duration-150",
              isOpen && "rotate-90",
            )}
            aria-hidden="true"
          />
          {tone ? (
            <span
              aria-hidden
              className="h-2 w-2 rounded-full shrink-0 border border-line"
              style={{ backgroundColor: tone.ink }}
            />
          ) : null}
        </span>
        <span className="inline-flex min-w-0 items-baseline gap-2">
          <span
            className={cn(
              "truncate text-sm font-bold text-ink-soft",
              isOpen ? "font-semibold text-ink" : "font-bold",
            )}
          >
            {node.name}
          </span>
          <span className="rounded-md shrink-0 border border-line bg-surface px-1.5 font-mono text-[10px] font-semibold tabular-nums text-ink">
            {node.count}
          </span>
        </span>
      </button>

      {isOpen ? (
        <div className="ml-[17px] mt-2 grid min-w-0 gap-1 overflow-hidden border-l border-line pl-3">
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
              className="group/file rounded-lg grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1.5 overflow-hidden border border-transparent px-2.5 py-2 text-[13px] font-semibold leading-snug text-ink-muted no-underline transition-colors hover:border-line hover:bg-recess hover:text-ink"
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
