import { Github, Heart, ArrowUpRight } from "lucide-react";
import { CSDN_URL, PROFILE_URL } from "@/lib/constants";
export function AboutCard() {
  return <section className="profile-card" aria-label="关于博主">
    <div className="profile-banner"><span>HELLO, FRIEND</span><Heart size={16} /></div>
    <div className="profile-body"><img src="./blog-icon.png" alt="Golemon 头像" width={60} height={60} className="profile-avatar" />
    <p className="profile-name">Golemon <span>✧</span></p><p className="profile-handle">@golitter · 技术笔记</p>
    <p className="profile-description">记录大模型、智能体与系统工程相关的学习与实践。</p>
    <div className="profile-links"><a href={PROFILE_URL} target="_blank" rel="noreferrer"><Github size={15} /> GitHub <ArrowUpRight size={13} /></a><a href={CSDN_URL} target="_blank" rel="noreferrer">CSDN · 田乐蒙 <ArrowUpRight size={13} /></a></div>
    </div><p className="profile-footer">慢慢来，每一小步都算数 ♡</p>
  </section>;
}
