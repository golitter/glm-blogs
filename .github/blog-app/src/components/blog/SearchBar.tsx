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
      <div className="flex h-14 items-center gap-3 border-[3px] border-line bg-primary px-3.5 shadow-soft transition-transform focus-within:-translate-y-0.5 focus-within:shadow-lift">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center border-2 border-ink bg-sun">
          <Search className="h-4 w-4 text-ink" aria-hidden />
        </span>
        <input
          ref={inputRef}
          className="peer h-full w-full bg-transparent text-[15px] font-bold text-white outline-none placeholder:text-white"
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
