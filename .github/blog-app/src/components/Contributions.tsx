import { useState } from "react";
import snapshot from "@/generated/contributions.json";
import { contributionProjects, type ContributionProject } from "@/lib/contributions";
import { PROFILE_URL } from "@/lib/constants";
import "./contributions.css";

const projects = contributionProjects(snapshot);
const total = projects.reduce((sum, project) => sum + project.pullRequests.length, 0);

function Project({ project, index }: { project: ContributionProject; index: number }) {
  const [expanded, setExpanded] = useState(true);
  const [owner, name] = project.fullName.split("/");
  const id = `contribution-project-${index}`;
  return <article className="contribution-project" aria-labelledby={`${id}-name`} data-stars={project.stars} data-tone={index % 5}>
    <header className="contribution-project-header">
      <h2 id={`${id}-name`}><a href={project.url} target="_blank" rel="noreferrer"><span className="contribution-owner">{owner} / </span>{name}</a></h2>
      <div className="contribution-project-numbers">
        <span className="contribution-stars" title={`${project.stars.toLocaleString("en-US")} Stars`}><span aria-hidden="true">☆</span>{project.stars.toLocaleString("en-US")}<span>Stars</span></span>
        <button className="contribution-toggle" type="button" aria-expanded={expanded} aria-controls={`${id}-prs`} aria-label={`${expanded ? "收起" : "展开"} ${name} 的 ${project.pullRequests.length} 个已合并 PR`} onClick={() => setExpanded(!expanded)}>{project.pullRequests.length} 个 PR<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg></button>
      </div>
    </header>
    <ul className="contribution-pr-list" id={`${id}-prs`} hidden={!expanded}>
      {project.pullRequests.map(pr => <li className="contribution-pr" key={pr.number}>
        <h3><a href={pr.url} target="_blank" rel="noreferrer"><span className="contribution-pr-number">#{pr.number}</span><span>{pr.title}</span></a></h3>
      </li>)}
    </ul>
  </article>;
}

export function Contributions() {
  return <div className="contributions-page">
    <a className="contribution-skip" href="#contributions-content" onClick={event => { event.preventDefault(); document.getElementById("contributions-content")?.focus(); }}>跳到贡献列表</a>
    <header className="contributions-site-header">
      <a className="contributions-back" href="#/" aria-label="返回博客首页"><span aria-hidden="true">←</span>返回博客</a>
      <a className="contributions-profile" href={PROFILE_URL} target="_blank" rel="noreferrer">@{snapshot.username}<span aria-hidden="true">↗</span></a>
    </header>
    <main className="contributions-content" id="contributions-content" tabIndex={-1}>
      <header className="contributions-heading">
        <h1 id="contributions-title">开源贡献</h1>
        <p><span className="contributions-total"><strong data-testid="merged-pr-count">{total}</strong> 个 PR 已合并</span><span>{projects.length} 个项目</span></p>
      </header>
      <p className="contributions-sort">项目按 Stars 从高到低排列</p>
      <div className="contributions-projects">{projects.map((project, index) => <Project project={project} index={index} key={project.fullName}/>)}</div>
      {!projects.length && <p className="contributions-empty">还没有已合并的公开 PR，新的贡献会在这里记录。</p>}
    </main>
  </div>;
}
