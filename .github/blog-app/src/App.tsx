import { useEffect, useState } from "react";
import { ImmersiveBlog } from "@/components/ImmersiveBlog";
import { Contributions } from "@/components/Contributions";
import { parseRoute } from "@/lib/blog-experience";

export default function App() {
  const [view, setView] = useState(() => parseRoute(location.hash).view);
  useEffect(() => {
    const change = () => setView(parseRoute(location.hash).view);
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  return view === "contributions" ? <Contributions /> : <ImmersiveBlog />;
}
