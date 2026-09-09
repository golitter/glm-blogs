import type { CSSProperties, PointerEvent, ReactNode } from "react";
import { useRef } from "react";

import { cn } from "@/lib/utils";

type TiltStyle = CSSProperties & {
  "--tilt-x": string;
  "--tilt-y": string;
  "--glow-x": string;
  "--glow-y": string;
};

export function TiltSurface({ children, className, strength = 5 }: {
  children: ReactNode;
  className?: string;
  strength?: number;
}) {
  const surface = useRef<HTMLDivElement>(null);

  function updateTilt(event: PointerEvent<HTMLDivElement>) {
    if (event.pointerType === "touch" || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const element = surface.current;
    if (!element) return;
    const rect = element.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    element.style.setProperty("--tilt-x", `${(0.5 - y) * strength}deg`);
    element.style.setProperty("--tilt-y", `${(x - 0.5) * strength}deg`);
    element.style.setProperty("--glow-x", `${x * 100}%`);
    element.style.setProperty("--glow-y", `${y * 100}%`);
  }

  function resetTilt() {
    const element = surface.current;
    if (!element) return;
    element.style.setProperty("--tilt-x", "0deg");
    element.style.setProperty("--tilt-y", "0deg");
    element.style.setProperty("--glow-x", "50%");
    element.style.setProperty("--glow-y", "0%");
  }

  const style: TiltStyle = {
    "--tilt-x": "0deg",
    "--tilt-y": "0deg",
    "--glow-x": "50%",
    "--glow-y": "0%",
  };

  return <div className={cn("tilt-stage", className)}>
    <div className="tilt-shadow" aria-hidden="true" />
    <div ref={surface} className="tilt-surface" style={style} onPointerMove={updateTilt} onPointerLeave={resetTilt}>
      <span className="tilt-back" aria-hidden="true" />
      <span className="tilt-edge tilt-edge-right" aria-hidden="true" />
      <span className="tilt-edge tilt-edge-bottom" aria-hidden="true" />
      <span className="tilt-shine" aria-hidden="true" />
      <div className="tilt-face">{children}</div>
    </div>
  </div>;
}
