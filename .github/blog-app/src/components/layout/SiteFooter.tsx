import { Feather } from "lucide-react";
import { updateTime } from "@/generated/blog-data";
export function SiteFooter() {
  return <footer className="site-footer"><div><p><Feather size={15} /> Golemon Blogs <span>· 小鸟的云端手账</span></p><small>内容托管于 GitHub，本页由仓库目录自动生成。</small></div><small>最后更新 {updateTime}（中国时间）</small></footer>;
}
