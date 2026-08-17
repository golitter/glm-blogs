import { updateTime } from "@/generated/blog-data";

export function SiteFooter() {
  return (
    <footer className="mt-14 flex items-end justify-between gap-6 border-t-[3px] border-line py-8 max-[560px]:block">
      <p className="text-xs font-bold leading-relaxed text-ink-soft">
        内容托管于 GitHub，本页由仓库目录自动生成。
      </p>
      <p className="border-2 border-ink bg-sun px-2 py-1 font-mono text-[10px] font-black tabular-nums text-ink shadow-[2px_2px_0_#111] max-[560px]:mt-4 max-[560px]:inline-block">
        最后更新 {updateTime}（中国时间）
      </p>
    </footer>
  );
}
