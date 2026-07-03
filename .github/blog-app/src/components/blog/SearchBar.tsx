import { Search } from "lucide-react";
import { type FormEvent, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";

export function SearchBar() {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleKeydown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, []);

  function performSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    const repoQuery = `repo:golitter/glm-blogs ${trimmed}`;
    window.open(`https://github.com/search?q=${encodeURIComponent(repoQuery)}&type=code`, "_blank");
  }

  return (
    <form onSubmit={performSearch} className="group">
      <div className="flex h-12 items-center gap-2.5 rounded-xl border border-line bg-surface px-3.5 shadow-soft transition-colors focus-within:border-line-strong focus-within:ring-2 focus-within:ring-ink/10">
        <Search className="h-4 w-4 shrink-0 text-ink-muted" aria-hidden />
        <input
          ref={inputRef}
          className="peer h-full w-full bg-transparent text-[15px] text-ink outline-none placeholder:text-ink-faint"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索笔记，例如 qwen、agent"
          autoComplete="off"
          aria-label="搜索笔记"
          required
        />
        <kbd className="hidden shrink-0 peer-focus:hidden sm:inline-flex">⌘K</kbd>
        <Button type="submit" className="h-8 shrink-0 rounded-md px-3 text-xs" aria-label="搜索">
          搜索
        </Button>
      </div>
    </form>
  );
}
