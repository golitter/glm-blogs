import { useState } from "react";

import { CategoryNode } from "@/components/blog/CategoryNode";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { blogTree } from "@/generated/blog-data";
import { toneFor } from "@/lib/category-style";
import { REPO_URL } from "@/lib/constants";

export function CategoryTree() {
  const [openPaths, setOpenPaths] = useState<Set<string>>(new Set());

  function togglePath(path: string) {
    setOpenPaths((current) => {
      const next = new Set(current);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }

  function collapseCategories() {
    setOpenPaths(new Set());
  }

  return (
    <Card className="animate-fade-in-up">
      <CardHeader className="flex flex-row items-center justify-between gap-3 pb-3">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faint">Contents</p>
          <CardTitle className="mt-1">
            <a className="no-underline hover:text-ink-soft" href={REPO_URL} target="_blank" rel="noreferrer">
              目录
            </a>
          </CardTitle>
        </div>
        <Button variant="ghost" size="sm" onClick={collapseCategories}>
          全部折叠
        </Button>
      </CardHeader>
      <CardContent className="grid gap-0.5">
        {blogTree.map((node, index) => (
          <CategoryNode
            key={node.path}
            node={node}
            openPaths={openPaths}
            togglePath={togglePath}
            tone={toneFor(index)}
          />
        ))}
      </CardContent>
    </Card>
  );
}
