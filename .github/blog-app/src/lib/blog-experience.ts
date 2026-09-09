import { pinyin } from "pinyin-pro";
import { blogTree, type BlogFile, type BlogTreeNode } from "../generated/blog-data";

export const flatten = (node: BlogTreeNode): BlogFile[] => [...node.files, ...node.children.flatMap(flatten)];
export const files = blogTree.flatMap(flatten);
export const nodes: BlogTreeNode[] = [];
function collect(items: BlogTreeNode[]) { for (const n of items) { nodes.push(n); collect(n.children); } }
collect(blogTree);
const normalize = (s: string) => s.normalize("NFKC").toLowerCase().replace(/[\s\u0300-\u036f]/g, "");
const index = files.map(file => {
  const text = `${file.title} ${file.path}`;
  return {file, text: normalize(text), phonetic: normalize(pinyin(text, {toneType: "none"})), initials: normalize(pinyin(text, {pattern: "first", toneType: "none"}))};
});
export function searchFiles(query: string) {
  const terms = query.trim().split(/\s+/).map(normalize).filter(Boolean);
  return index.filter(item => terms.every(q => item.text.includes(q) || item.phonetic.includes(q) || item.initials.includes(q))).map(item => item.file);
}
export function highlightedParts(text: string, query: string) {
  const chars = Array.from(text), hits = new Set<number>();
  for (const term of query.trim().split(/\s+/).map(normalize).filter(Boolean)) {
    for (const mode of ["text", "pinyin", "initials"]) {
      const positions: number[] = []; let joined = "";
      chars.forEach((char,i) => {
        const part = mode === "text" ? normalize(char) : normalize(pinyin(char, {toneType:"none", ...(mode === "initials" ? {pattern:"first" as const} : {})}));
        joined += part; positions.push(...Array(part.length).fill(i));
      });
      for (let at = joined.indexOf(term); at >= 0; at = joined.indexOf(term, at + term.length)) {
        for (let i = at; i < at + term.length; i++) hits.add(positions[i]);
      }
    }
  }
  return chars.map((char,i) => ({text:char, hit:hits.has(i)}));
}
export function suggestions(query: string) {
  const q = normalize(query);
  const score = (node: BlogTreeNode) => {
    const name = normalize(node.path);
    return [...new Set(q)].filter(char => name.includes(char)).length / Math.max(1, new Set(q + name).size);
  };
  return [...nodes].sort((a,b) => score(b) - score(a) || b.count - a.count).slice(0, 3);
}
export type Quality = "high" | "balanced" | "eco";
export type Route = {view: "home" | "shelf" | "recent" | "search" | "category"; path?: string; q?: string; camera?: number[]};
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
    const route: Route = {view: ["shelf", "recent", "search", "category"].includes(view) ? view as Route["view"] : "home"};
    if (route.view === "category") route.path = parts.map(decodeURIComponent).join("/");
    if (route.view === "search") route.q = params.get("q") ?? "";
    const camera = params.get("camera")?.split(",").map(Number);
    if (camera?.length === 6 && camera.every(n => Number.isFinite(n) && Math.abs(n) < 10000)) route.camera = camera;
    return route;
  } catch { return {view: "home"}; }
}
export function readLocal<T>(key: string, fallback: T): T { try { return JSON.parse(localStorage.getItem(key) ?? "null") ?? fallback; } catch { return fallback; } }
export function writeLocal(key: string, value: unknown) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Private storage is optional. */ } }
export function updateInfo(file: BlogFile, now = Date.now()) {
  const age = file.updatedAt ? Math.max(0, now - file.updatedAt) : Infinity;
  const today = new Date(now); today.setHours(0,0,0,0);
  const monday = new Date(today); monday.setDate(today.getDate() - (today.getDay() + 6) % 7);
  const bucket = !file.updatedAt ? "时间未知" : file.updatedAt >= +today ? "今天" : file.updatedAt >= +monday ? "本周" : "更早";
  const days = Math.floor(age / 86400000);
  const relative = !Number.isFinite(age) ? "暂无记录" : age < 3600000 ? "刚刚" : age < 86400000 ? `${Math.floor(age / 3600000)} 小时前` : `${days} 天前`;
  return {bucket, relative, isNew: file.change === "added" && age < 7 * 86400000, age};
}
