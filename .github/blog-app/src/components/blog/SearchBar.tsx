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
    if (!query.trim()) event.preventDefault();
  }

  return (
    <form action="https://github.com/search" method="get" target="_blank" rel="noopener noreferrer" onSubmit={performSearch} className="group search-form">
      <input type="hidden" name="q" value={`repo:golitter/glm-blogs ${query.trim()}`} />
      <input type="hidden" name="type" value="code" />
      <div className="rounded-2xl flex h-16 items-center gap-3 border border-line bg-surface px-3.5 shadow-soft transition-transform focus-within:-translate-y-0.5 focus-within:shadow-lift">
        <span className="rounded-full flex h-8 w-8 shrink-0 items-center justify-center border border-line bg-sun">
          <Search className="h-4 w-4 text-ink" aria-hidden />
        </span>
        <input
          ref={inputRef}
          type="search"
          className="peer h-full w-full bg-transparent text-[15px] font-normal text-ink outline-none placeholder:text-ink-muted"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索笔记，例如 qwen、agent"
          autoComplete="off"
          aria-label="搜索笔记"
          required
        />
        <kbd className="search-shortcut shrink-0">Ctrl K</kbd>
        <Button type="submit" className="h-9 shrink-0 bg-sun px-4 text-xs text-ink hover:bg-recess" aria-label="搜索">
          搜索
        </Button>
      </div>
    </form>
  );
}
