import { ChevronRight } from "lucide-react";

import type { BlogTreeNode } from "@/generated/blog-data";
import type { CategoryTone } from "@/lib/category-style";
import { cn } from "@/lib/utils";

export function CategoryNode({
  node,
  openPaths,
  togglePath,
  tone,
}: {
  node: BlogTreeNode;
  openPaths: Set<string>;
  togglePath: (path: string) => void;
  tone?: CategoryTone;
}) {
  const isOpen = openPaths.has(node.path);

  return (
    <div className="rounded-md">
      <button
        type="button"
        className={cn(
          "grid min-h-9 w-full grid-cols-[auto_minmax(0,1fr)] items-center gap-2 rounded-md px-2 py-1.5 text-left transition-colors hover:bg-recess active:scale-[0.995]",
          isOpen && "bg-recess",
        )}
        onClick={() => togglePath(node.path)}
      >
        <span className="inline-flex items-center gap-2 text-ink">
          <ChevronRight
            className={cn(
              "h-3.5 w-3.5 shrink-0 text-ink-faint transition-transform duration-200",
              isOpen && "rotate-90",
            )}
            aria-hidden="true"
          />
          {tone ? (
            <span
              aria-hidden
              className="h-2 w-2 shrink-0 rounded-[3px]"
              style={{ backgroundColor: tone.ink }}
            />
          ) : null}
        </span>
        <span className="inline-flex min-w-0 items-baseline gap-2">
          <span
            className={cn(
              "truncate text-sm text-ink-soft",
              isOpen ? "font-semibold text-ink" : "font-medium",
            )}
          >
            {node.name}
          </span>
          <span className="shrink-0 font-mono text-[11px] tabular-nums text-ink-faint">
            {node.count}
          </span>
        </span>
      </button>

      {isOpen ? (
        <div className="ml-[15px] grid min-w-0 gap-px overflow-hidden border-l border-line pl-3">
          {node.children.map((child) => (
            <CategoryNode key={child.path} node={child} openPaths={openPaths} togglePath={togglePath} />
          ))}
          {node.files.map((file) => (
            <a
              key={file.path}
              href={file.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group/file grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1.5 overflow-hidden rounded-md px-2.5 py-1.5 text-[13px] font-medium leading-snug text-ink-muted no-underline transition-colors hover:bg-recess hover:text-ink"
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
