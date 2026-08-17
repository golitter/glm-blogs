import type { ReactNode } from "react";

import { markdownCount } from "@/generated/blog-data";
import { REPO_LINKS } from "@/lib/constants";

function NavLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      className="inline-flex min-h-11 items-center border-2 border-transparent px-2.5 py-1.5 text-xs font-black text-ink no-underline transition-colors hover:border-ink hover:bg-sun max-[560px]:px-1.5 max-[560px]:text-[11px]"
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
    <header className="sticky top-0 z-10 border-b-[3px] border-ink bg-surface">
      <nav className="mx-auto flex min-h-20 w-full max-w-[1320px] items-center justify-between gap-5 px-6 max-[560px]:min-h-16 max-[560px]:px-4">
        <div className="flex min-w-0 flex-wrap items-center gap-3">
          <a
            className="inline-flex items-center gap-3 text-lg font-black uppercase tracking-[-0.035em] no-underline text-ink"
            href="./"
          >
            <img
              className="h-10 w-10 border-2 border-line bg-sun object-cover shadow-[3px_3px_0_#111] max-[560px]:h-8 max-[560px]:w-8"
              src="./blog-icon.png"
              alt="Golemon Blogs"
              width={40}
              height={40}
              decoding="async"
            />
            <span className="max-[420px]:hidden">Golemon Blogs</span>
          </a>
          <span className="inline-flex items-baseline gap-1.5 border-2 border-line bg-sun px-2 py-1.5 font-mono text-[10px] font-bold leading-none text-ink">
            <strong className="text-[12px] font-black tabular-nums text-ink">{markdownCount}</strong>
            篇笔记
          </span>
        </div>

        <div className="flex items-center gap-0.5">
          <NavLink href={REPO_LINKS.home}>仓库</NavLink>
          <NavLink href={REPO_LINKS.commits}>提交</NavLink>
          <NavLink href={REPO_LINKS.issues}>问题</NavLink>
        </div>
      </nav>
    </header>
  );
}
