import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { writeFile, rename, rm } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { PROFILE_URL } from "../src/lib/constants.ts";
import { contributionProjects, validateContributionSnapshot } from "../src/lib/contributions.ts";

const exec = promisify(execFile);

async function github(path) {
  const token = process.env.GH_TOKEN || process.env.GITHUB_TOKEN;
  // The CLI uses the user's existing login locally; Actions can use its environment token.
  // Neither credentials nor raw authenticated responses are written into the snapshot.
  if (!token) {
    try {
      const { stdout } = await exec("gh", ["api", "--method", "GET", path], {
        timeout: 40_000, maxBuffer: 16 * 1024 * 1024,
      });
      return JSON.parse(stdout);
    } catch (error) {
      // gh exit code 4 means authentication is required; public APIs can still work.
      if (error.code !== "ENOENT" && error.code !== 4) throw error;
    }
  }
  const response = await fetch(`https://api.github.com/${path}`, {
    headers: {
      Accept: "application/vnd.github+json",
      "User-Agent": "golemon-blogs-contributions",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    signal: AbortSignal.timeout(40_000),
  });
  if (!response.ok) throw new Error(`GitHub returned HTTP ${response.status} for ${path.split("?")[0]}`);
  return response.json();
}

async function mapConcurrent(items, transform) {
  const results = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(3, items.length) }, async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await transform(items[index]);
    }
  }));
  return results;
}

export async function syncContributions({
  username = new URL(PROFILE_URL).pathname.split("/").filter(Boolean)[0],
  outputPath = new URL("../src/generated/contributions.json", import.meta.url),
  githubRequest = github,
  now = new Date(),
} = {}) {
  const items = [];
  let total = 0;
  for (let page = 1; ; page++) {
    const query = new URLSearchParams({
      q: `author:${username} is:pr is:merged is:public`,
      per_page: "100", page: String(page), sort: "created", order: "asc",
    });
    const response = await githubRequest(`search/issues?${query}`);
    if (!response || !Number.isSafeInteger(response.total_count) || response.total_count < 0
      || typeof response.incomplete_results !== "boolean" || !Array.isArray(response.items)) {
      throw new Error("Invalid GitHub search response. Existing data was preserved.");
    }
    if (response.incomplete_results) throw new Error("GitHub returned incomplete search results. Existing data was preserved.");
    if (page === 1) total = response.total_count;
    else if (response.total_count !== total) throw new Error("PR search changed during pagination. Retry the sync.");
    if (total > 1000) throw new Error("More than 1,000 PRs: split the search by merge date before syncing.");
    items.push(...response.items);
    if (items.length > total) throw new Error("PR search returned more records than expected. Existing data was preserved.");
    if (items.length === total) break;
    if (!response.items.length) throw new Error("PR pagination ended early. Existing data was preserved.");
  }

  // Verify each result against the PR endpoint, rather than treating closed issues as merges.
  const pullRequests = await mapConcurrent(items, async item => {
    const url = new URL(item.pull_request?.url);
    if (url.origin !== "https://api.github.com" || !/^\/repos\/[a-z\d-]+\/[a-z\d_.-]+\/pulls\/[1-9]\d*$/i.test(url.pathname)
      || url.username || url.password || url.search || url.hash) throw new Error("Unexpected PR API URL");
    const pr = await githubRequest(url.pathname.slice(1));
    if (pr?.merged !== true || !pr.merged_at || pr.base?.repo?.private !== false
      || typeof pr.user?.login !== "string" || pr.user.login.toLowerCase() !== username.toLowerCase()
      || pr.number !== Number(url.pathname.split("/").at(-1))) {
      throw new Error("PR did not pass public author/merge verification. Existing data was preserved.");
    }
    return {
      repository: pr.base.repo.full_name, number: pr.number, title: pr.title,
      url: pr.html_url, author: pr.user.login, merged: true, mergedAt: pr.merged_at,
    };
  });

  const repositoryNames = [...new Set(pullRequests.map(pr => pr.repository))].sort();
  const repositories = await mapConcurrent(repositoryNames, async fullName => {
    const repo = await githubRequest(`repos/${fullName}`);
    if (repo?.private !== false || typeof repo.full_name !== "string" || repo.full_name.toLowerCase() !== fullName.toLowerCase()) {
      throw new Error(`Invalid public repository data: ${fullName}`);
    }
    return {
      fullName: repo.full_name, name: repo.name, description: repo.description ?? "",
      url: repo.html_url, stars: repo.stargazers_count,
    };
  });

  const snapshot = validateContributionSnapshot({ username, syncedAt: now.toISOString(), repositories, pullRequests }, username, now);
  if (snapshot.pullRequests.length !== total) throw new Error(`Expected ${total} merged PRs, verified ${snapshot.pullRequests.length}. Existing data was preserved.`);

  // Replace only after every page, merge status and repository count has been verified.
  const output = outputPath instanceof URL ? fileURLToPath(outputPath) : resolve(outputPath);
  const temporary = `${output}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, JSON.stringify(snapshot, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
    await rename(temporary, output);
  } finally {
    await rm(temporary, { force: true });
  }
  return snapshot;
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  syncContributions().then(snapshot => {
    const projects = contributionProjects(snapshot);
    console.log(`Synced ${snapshot.pullRequests.length} merged PRs across ${projects.length} projects for @${snapshot.username}.`);
    for (const project of projects) console.log(`${project.fullName}: ${project.stars} stars, ${project.pullRequests.length} merged PRs`);
  }).catch(error => {
    console.error(`Contribution sync failed: ${error.message}`);
    process.exitCode = 1;
  });
}
