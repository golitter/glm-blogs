import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { recentFiles } from "@/generated/blog-data";

export function RecentUpdates() {
  const latestFiles = recentFiles.slice(0, 5);

  return (
    <Card className="animate-fade-in-up" style={{ animationDelay: "80ms" }}>
      <CardHeader className="pb-3">
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faint">Latest</p>
        <CardTitle className="mt-1">最近更新</CardTitle>
      </CardHeader>
      <CardContent className="grid">
        {latestFiles.map((file, index) => (
          <a
            key={`${file.path}-${file.date}`}
            className="group flex items-start gap-3 border-line py-3 no-underline transition-colors first:border-t-0 [&+&]:border-t"
            href={file.url}
            target="_blank"
            rel="noreferrer"
          >
            <span className="mt-0.5 w-5 shrink-0 font-mono text-[11px] tabular-nums text-ink-faint">
              {String(index + 1).padStart(2, "0")}
            </span>
            <span className="min-w-0 flex-1">
              <span className="flex items-center gap-2">
                <span className="block text-balance text-[14px] font-semibold leading-snug text-ink transition-colors group-hover:text-ink-soft">
                  {file.title}
                </span>
                {index === 0 ? (
                  <span className="rounded-[3px] bg-tag-sage-bg px-1.5 py-px font-mono text-[9px] font-semibold uppercase tracking-wider text-tag-sage-ink">
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
