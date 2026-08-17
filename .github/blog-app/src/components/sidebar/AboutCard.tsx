import { Github } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { CSDN_URL, PROFILE_URL } from "@/lib/constants";

export function AboutCard() {
  return (
    <Card className="bg-signal">
      <CardContent className="p-5 text-ink">
        <div className="flex items-center gap-3">
          <img
            className="h-12 w-12 border-2 border-ink bg-sun object-cover shadow-[3px_3px_0_#111]"
            src="./blog-icon.png"
            alt="Golemon"
            width={44}
            height={44}
            decoding="async"
          />
          <div className="min-w-0">
            <p className="text-base font-black uppercase text-ink">Golemon</p>
            <p className="truncate text-xs font-semibold text-ink-soft">@golitter · 技术笔记</p>
          </div>
        </div>
        <p className="mt-4 border-y-2 border-ink py-3 text-[13px] font-semibold leading-relaxed text-ink">
          记录大模型、智能体与系统工程相关的学习与实践。
        </p>
        <div className="mt-4 flex flex-col gap-2">
          <a
            className="inline-flex w-full items-center justify-center gap-2 border-2 border-ink bg-ink px-3 py-2 text-xs font-black uppercase no-underline shadow-[3px_3px_0_#111] transition-transform hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[2px_2px_0_#111]"
            style={{ color: "#ffffff", backgroundColor: "#111111" }}
            href={PROFILE_URL}
            target="_blank"
            rel="noreferrer"
          >
            <Github className="h-4 w-4" aria-hidden />
            @golitter
          </a>
          <a
            className="inline-flex w-full items-center justify-center gap-2 border-2 border-ink bg-sun px-3 py-2 text-xs font-black uppercase text-ink no-underline shadow-[3px_3px_0_#111] transition-transform hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[2px_2px_0_#111]"
            style={{ color: "#111111" }}
            href={CSDN_URL}
            target="_blank"
            rel="noreferrer"
          >
            <span className="font-mono text-[10px] font-black tracking-wide text-ink">CSDN</span>
            @田乐蒙
          </a>
        </div>
      </CardContent>
    </Card>
  );
}
