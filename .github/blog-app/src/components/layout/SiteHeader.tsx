import type { ReactNode } from "react";

import { markdownCount } from "@/generated/blog-data";
import { REPO_LINKS } from "@/lib/constants";

function NavLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      className="rounded-md px-2 py-1.5 text-sm text-ink-muted no-underline transition-colors hover:bg-recess hover:text-ink"
      href={href}
      target="_blank"
      rel="noreferrer"
    >
      {children}
    </a>
  );
}

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-10 border-b border-line/80 bg-surface/85 backdrop-blur-md">
      <nav className="mx-auto flex min-h-16 w-full max-w-[1240px] items-center justify-between gap-6 px-6">
        <div className="flex min-w-0 flex-wrap items-center gap-3">
          <a
            className="inline-flex items-center gap-2.5 text-base font-semibold tracking-tight no-underline text-ink"
            href="./"
          >
            <img
              className="h-7 w-7 rounded-full border border-line bg-recess object-cover"
              src="./blog-icon.png"
              alt="Golemon Blogs"
              width={28}
              height={28}
              decoding="async"
            />
            <span>Golemon Blogs</span>
          </a>
          <span className="inline-flex items-baseline gap-1.5 rounded-md border border-line bg-surface px-2 py-1 font-mono text-[11px] leading-none text-ink-muted">
            <strong className="text-[12px] font-semibold tabular-nums text-ink">{markdownCount}</strong>
            notes
          </span>
        </div>

        <div className="flex items-center gap-0.5">
          <NavLink href={REPO_LINKS.home}>Repository</NavLink>
          <NavLink href={REPO_LINKS.commits}>Commits</NavLink>
          <NavLink href={REPO_LINKS.issues}>Issues</NavLink>
        </div>
      </nav>
    </header>
  );
}
