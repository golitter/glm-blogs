import { ArrowUp } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function BackToTop() {
  const [show, setShow] = useState(false);

  useEffect(() => {
    const onScroll = () => setShow(window.scrollY > 360);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <Button
      variant="default"
      className={cn(
        "fixed bottom-6 right-6 z-20 h-11 w-11 bg-signal p-0 text-white shadow-lift transition-[opacity,transform] duration-200 active:translate-x-2 active:translate-y-2 active:shadow-none max-[560px]:bottom-4 max-[560px]:right-4 max-[560px]:h-10 max-[560px]:w-10",
        show ? "translate-y-0 opacity-100" : "pointer-events-none translate-y-2 opacity-0",
      )}
      aria-label="回到顶部"
      onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
    >
      <ArrowUp className="h-4 w-4" />
    </Button>
  );
}
