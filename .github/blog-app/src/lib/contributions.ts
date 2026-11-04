export type ContributionRepository = {
  fullName: string;
  name: string;
  description: string;
  url: string;
  stars: number;
};

export type ContributionPullRequest = {
  repository: string;
  number: number;
  title: string;
  url: string;
  author: string;
  merged: boolean;
  mergedAt: string | null;
};

export type ContributionSnapshot = {
  username: string;
  syncedAt: string;
  repositories: ContributionRepository[];
  pullRequests: ContributionPullRequest[];
};

export type ContributionProject = ContributionRepository & {
  pullRequests: (ContributionPullRequest & { mergedAt: string })[];
};

const CLOCK_SKEW_MS = 5 * 60 * 1000;
const repositoryName = /^[a-z\d-]+\/[a-z\d_.-]+$/i;
const timestamp = (value: unknown) => typeof value === "string" ? Date.parse(value) : NaN;
const record = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;
const githubLink = (value: unknown, path: string) => typeof value === "string"
  && value.toLowerCase() === `https://github.com/${path}`.toLowerCase();

// Both the collector and the published-snapshot restore must enforce the same contract.
// Do not use the display filter as validation: it can silently discard invalid records.
export function validateContributionSnapshot(value: unknown, username: string, now = new Date()): ContributionSnapshot {
  if (!record(value) || typeof value.username !== "string" || typeof username !== "string" || !username
    || value.username.toLowerCase() !== username.toLowerCase()
    || !Array.isArray(value.repositories) || !Array.isArray(value.pullRequests)) {
    throw new Error("Invalid contribution snapshot or account");
  }
  const syncedAt = timestamp(value.syncedAt);
  if (!Number.isFinite(syncedAt) || !Number.isFinite(now.getTime()) || syncedAt > now.getTime() + CLOCK_SKEW_MS) {
    throw new Error("Invalid or future contribution sync date");
  }

  const repositories = new Set<string>();
  for (const repo of value.repositories) {
    if (!record(repo) || typeof repo.fullName !== "string" || !repositoryName.test(repo.fullName)
      || typeof repo.name !== "string" || repo.name.toLowerCase() !== repo.fullName.split("/")[1].toLowerCase()
      || [".", ".."].includes(repo.name) || typeof repo.description !== "string"
      || !githubLink(repo.url, repo.fullName) || typeof repo.stars !== "number" || !Number.isSafeInteger(repo.stars) || repo.stars < 0) {
      throw new Error("Invalid contribution repository fields or GitHub link");
    }
    const key = repo.fullName.toLowerCase();
    if (repositories.has(key)) throw new Error("Duplicate contribution repository");
    repositories.add(key);
  }

  const seen = new Set<string>();
  const usedRepositories = new Set<string>();
  for (const pr of value.pullRequests) {
    if (!record(pr) || typeof pr.repository !== "string" || !repositories.has(pr.repository.toLowerCase())
      || typeof pr.number !== "number" || !Number.isSafeInteger(pr.number) || pr.number < 1
      || typeof pr.title !== "string" || !pr.title.trim() || typeof pr.author !== "string"
      || pr.author.toLowerCase() !== username.toLowerCase() || pr.merged !== true
      || !Number.isFinite(timestamp(pr.mergedAt)) || timestamp(pr.mergedAt) > syncedAt + CLOCK_SKEW_MS
      || !githubLink(pr.url, `${pr.repository}/pull/${pr.number}`)) {
      throw new Error("Invalid, unmerged or unrelated contribution PR");
    }
    const repositoryKey = pr.repository.toLowerCase();
    const key = `${repositoryKey}#${pr.number}`;
    if (seen.has(key)) throw new Error("Duplicate contribution PR");
    seen.add(key);
    usedRepositories.add(repositoryKey);
  }
  if (usedRepositories.size !== repositories.size) throw new Error("Contribution repository has no merged PRs");
  return value as ContributionSnapshot;
}

export function contributionProjects(snapshot: ContributionSnapshot): ContributionProject[] {
  const repositories = new Map(snapshot.repositories.map(repo => [repo.fullName.toLowerCase(), repo]));
  const projects = new Map<string, ContributionProject>();
  const seen = new Set<string>();

  for (const pr of snapshot.pullRequests) {
    // A closed PR is not necessarily merged. Require the explicit merge flag and timestamp.
    if (pr.merged !== true || !pr.mergedAt || !Number.isFinite(Date.parse(pr.mergedAt))) continue;
    if (pr.author.toLowerCase() !== snapshot.username.toLowerCase()) continue;
    const repositoryKey = pr.repository.toLowerCase();
    const repo = repositories.get(repositoryKey);
    const key = `${repositoryKey}#${pr.number}`;
    if (!repo || seen.has(key)) continue;
    seen.add(key);
    let project = projects.get(repositoryKey);
    if (!project) {
      project = { ...repo, pullRequests: [] };
      projects.set(repositoryKey, project);
    }
    project.pullRequests.push({ ...pr, mergedAt: pr.mergedAt });
  }

  for (const project of projects.values()) {
    project.pullRequests.sort((a, b) => Date.parse(b.mergedAt) - Date.parse(a.mergedAt) || b.number - a.number);
  }
  return [...projects.values()].sort((a, b) => b.stars - a.stars || a.fullName.localeCompare(b.fullName));
}
