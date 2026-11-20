export type Route = {view: "home" | "shelf" | "recent" | "search" | "category" | "contributions"; path?: string; q?: string; camera?: number[]};
export function routeHash(route: Route) {
  const params = new URLSearchParams();
  if (route.q) params.set("q", route.q);
  if (route.camera) params.set("camera", route.camera.map(n => n.toFixed(2)).join(","));
  return `#/${route.view === "home" ? "" : route.view}${route.view === "category" ? "/" + (route.path ?? "").split("/").map(encodeURIComponent).join("/") : ""}${params.size ? "?" + params : ""}`;
}
export function parseRoute(hash: string): Route {
  try {
    const [raw, query] = hash.replace(/^#\/?/, "").split("?");
    const [view, ...parts] = raw.split("/");
    const params = new URLSearchParams(query);
    const route: Route = {view: ["shelf", "recent", "search", "category", "contributions"].includes(view) ? view as Route["view"] : "home"};
    if (route.view === "category") route.path = parts.map(decodeURIComponent).join("/");
    if (route.view === "search") route.q = params.get("q") ?? "";
    const camera = params.get("camera")?.split(",").map(Number);
    if (camera?.length === 6 && camera.every(n => Number.isFinite(n) && Math.abs(n) < 10000)) route.camera = camera;
    return route;
  } catch { return {view: "home"}; }
}
