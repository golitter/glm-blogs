import type { ReactNode } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function Chip({ children }: { children: ReactNode }) {
  return (
    <code className="border-2 border-line bg-surface px-1.5 py-0.5 font-mono text-[11px] font-bold text-ink shadow-[2px_2px_0_#111]">
      {children}
    </code>
  );
}

export function SearchTips() {
  return (
    <Card className="bg-sun">
      <CardHeader className="pb-2">
        <p className="font-mono text-[10px] font-black tracking-[0.2em] text-ink">搜索帮助</p>
        <CardTitle className="mt-1">搜索语法</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 border-t-2 border-ink pt-4 text-[13px] font-semibold leading-relaxed text-ink-soft">
        <p>关键词含空格时，用英文双引号包裹；多个关键词用逗号分隔。</p>
        <ul className="space-y-2">
          <li className="flex items-center gap-2">
            <Chip>"go context"</Chip>
            <span className="text-ink-soft">精确短语</span>
          </li>
          <li className="flex items-center gap-2">
            <Chip>agent, rag</Chip>
            <span className="text-ink-soft">多关键词</span>
          </li>
        </ul>
        <p className="flex items-center gap-1.5 pt-1 text-ink-soft">
          按
          <kbd>Ctrl</kbd>
          <kbd>K</kbd>
          快速聚焦搜索框。
        </p>
      </CardContent>
    </Card>
  );
}
