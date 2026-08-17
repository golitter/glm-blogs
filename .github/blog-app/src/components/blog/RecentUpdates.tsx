import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { recentFiles } from "@/generated/blog-data";

export function RecentUpdates() {
  const latestFiles = recentFiles.slice(0, 5);

  return (
    <Card className="animate-fade-in-up overflow-hidden bg-surface" style={{ animationDelay: "80ms" }}>
      <CardHeader className="border-b-[3px] border-ink bg-primary pb-4 text-white">
        <p className="font-mono text-[10px] font-black tracking-[0.2em] text-white">最近发布</p>
        <CardTitle className="mt-1 text-white">最近更新</CardTitle>
      </CardHeader>
      <CardContent className="grid p-0">
        {latestFiles.map((file, index) => (
          <a
            key={`${file.path}-${file.date}`}
            className="group flex items-start gap-3 border-line p-4 no-underline transition-colors hover:bg-sun [&+&]:border-t-2"
            href={file.url}
            target="_blank"
            rel="noreferrer"
          >
            <span className="mt-0.5 flex h-6 w-7 shrink-0 items-center justify-center border-2 border-ink bg-signal font-mono text-[10px] font-black tabular-nums text-white">
              {String(index + 1).padStart(2, "0")}
            </span>
            <span className="min-w-0 flex-1">
              <span className="flex items-center gap-2">
                <span className="block text-balance text-[14px] font-black leading-snug text-ink">
                  {file.title}
                </span>
                {index === 0 ? (
                  <span className="border border-ink bg-mint px-1.5 py-px font-mono text-[9px] font-black uppercase tracking-wider text-ink">
                    New
                  </span>
                ) : null}
              </span>
              <span className="mt-1 block break-all font-mono text-[11px] leading-snug text-ink-faint">
                {file.path}
              </span>
              <span className="mt-1.5 block font-mono text-[11px] tabular-nums text-ink-muted">
                {file.date}
              </span>
            </span>
            <span
              aria-hidden
              className="mt-0.5 shrink-0 text-ink-faint opacity-0 transition-opacity group-hover:opacity-100"
            >
              ↗
            </span>
          </a>
        ))}
      </CardContent>
    </Card>
  );
}
