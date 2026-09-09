import { Bird, ArrowUpRight } from "lucide-react";
import { REPO_LINKS } from "@/lib/constants";
export function SiteHeader() {
  return <header className="site-header"><nav className="site-shell nav-inner" aria-label="主导航">
    <a className="brand" href="./"><span className="brand-mark"><Bird size={25} strokeWidth={1.5} /></span><span>Golemon <strong>Blogs</strong><small>A SOFT PLACE FOR IDEAS</small></span></a>
    <div className="nav-links"><a href="#notes">笔记本</a><a href="#recent">最近更新</a><span className="nav-separator" /><a href={REPO_LINKS.home} target="_blank" rel="noreferrer">GitHub <ArrowUpRight size={14} /></a><a className="nav-secondary" href={REPO_LINKS.commits} target="_blank" rel="noreferrer">提交</a><a className="nav-secondary" href={REPO_LINKS.issues} target="_blank" rel="noreferrer">问题</a></div>
  </nav></header>;
}
