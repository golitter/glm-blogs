import type { ReactNode } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function Chip({ children }: { children: ReactNode }) {
  return (
    <code className="rounded-[5px] border border-line bg-recess px-1.5 py-0.5 font-mono text-[11px] text-ink-soft">
      {children}
    </code>
  );
}

export function SearchTips() {
  return (
    <Card>
      <CardHeader className="pb-2">
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faint">Syntax</p>
        <CardTitle className="mt-1">搜索语法</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-[13px] leading-relaxed text-ink-soft">
        <p>关键词含空格时，用英文双引号包裹；多个关键词用逗号分隔。</p>
        <ul className="space-y-2">
          <li className="flex items-center gap-2">
            <Chip>"go context"</Chip>
            <span className="text-ink-muted">精确短语</span>
          </li>
          <li className="flex items-center gap-2">
            <Chip>agent, rag</Chip>
            <span className="text-ink-muted">多关键词</span>
          </li>
        </ul>
        <p className="flex items-center gap-1.5 pt-1 text-ink-muted">
          按
          <kbd>⌘</kbd>
          <kbd>K</kbd>
          快速聚焦搜索框。
        </p>
      </CardContent>
    </Card>
  );
}
