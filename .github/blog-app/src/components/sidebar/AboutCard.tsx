import { Github } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { CSDN_URL, PROFILE_URL } from "@/lib/constants";

export function AboutCard() {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-center gap-3">
          <img
            className="h-11 w-11 rounded-full border border-line bg-recess object-cover"
            src="./blog-icon.png"
            alt="Golemon"
            width={44}
            height={44}
            decoding="async"
          />
          <div className="min-w-0">
            <p className="text-sm font-semibold text-ink">Golemon</p>
            <p className="truncate text-xs text-ink-muted">@golitter · 技术笔记</p>
          </div>
        </div>
        <p className="mt-3 text-[13px] leading-relaxed text-ink-soft">
          记录大模型、智能体与系统工程相关的学习与实践。
        </p>
        <div className="mt-4 flex flex-col gap-2">
          <a
            className="inline-flex w-full items-center justify-center gap-2 rounded-md border border-line bg-surface px-3 py-2 text-xs font-medium text-ink-soft no-underline transition-colors hover:border-line-strong hover:bg-recess hover:text-ink"
            href={PROFILE_URL}
            target="_blank"
            rel="noreferrer"
          >
            <Github className="h-4 w-4" aria-hidden />
            @golitter
          </a>
          <a
            className="inline-flex w-full items-center justify-center gap-2 rounded-md border border-line bg-surface px-3 py-2 text-xs font-medium text-ink-soft no-underline transition-colors hover:border-line-strong hover:bg-recess hover:text-ink"
            href={CSDN_URL}
            target="_blank"
            rel="noreferrer"
          >
            <span className="font-mono text-[10px] font-bold tracking-wide text-ink-muted">CSDN</span>
            @田乐蒙
          </a>
        </div>
      </CardContent>
    </Card>
  );
}
