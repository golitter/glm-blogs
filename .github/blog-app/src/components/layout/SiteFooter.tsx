import { updateTime } from "@/generated/blog-data";

export function SiteFooter() {
  return (
    <footer className="mt-12 border-t border-line py-8">
      <p className="text-xs leading-relaxed text-ink-muted">
        内容托管于 GitHub，本页由仓库目录自动生成。
      </p>
      <p className="mt-3 font-mono text-[11px] tabular-nums text-ink-faint">
        最后更新 {updateTime}（中国时间）
      </p>
    </footer>
  );
}
