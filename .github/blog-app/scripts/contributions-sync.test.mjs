import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setImmediate } from "node:timers/promises";
import { test } from "node:test";
import { syncContributions } from "./sync-contributions.mjs";
import { contributionProjects } from "../src/lib/contributions.ts";

const now = new Date("2026-11-04T12:00:00Z");
const fullName = number => `org/${number % 2 ? "small" : "large"}`;
const previous = '{"sentinel":"previous snapshot"}\n';
async function fixture(t) {
  const directory = await mkdtemp(join(tmpdir(), "glm-contribution-sync-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const outputPath = join(directory, "contributions.json");
  await writeFile(outputPath, previous);
  return { outputPath, now };
}

function githubFixture({ total = 1, change = value => value } = {}) {
  const calls = [];
  let active = 0;
  let peak = 0;
  const githubRequest = async request => {
    calls.push(request);
    active++;
    peak = Math.max(peak, active);
    try {
      await setImmediate();
      const url = new URL(request, "https://api.github.com/");
      let data;
      if (url.pathname === "/search/issues") {
        assert.equal(url.searchParams.get("q"), "author:golitter is:pr is:merged is:public");
        const start = (Number(url.searchParams.get("page")) - 1) * 100;
        data = { total_count: total, incomplete_results: false, items: Array.from({ length: Math.min(100, total - start) }, (_, i) => {
          const number = start + i + 1;
          return { pull_request: { url: `https://api.github.com/repos/${fullName(number)}/pulls/${number}` } };
        }) };
      } else if (url.pathname.includes("/pulls/")) {
        const number = Number(url.pathname.split("/").at(-1));
        data = {
          number, title: `Fixture PR ${number}`, html_url: `https://github.com/${fullName(number)}/pull/${number}`,
          merged: true, merged_at: "2026-11-04T02:00:00Z", user: { login: "golitter" },
          base: { repo: { full_name: fullName(number), private: false } },
        };
      } else {
        const name = url.pathname.replace("/repos/", "");
        data = { full_name: name, name: name.split("/")[1], private: false, description: null,
          html_url: `https://github.com/${name}`, stargazers_count: name.endsWith("large") ? 1000 : 5 };
      }
      return change(data, url);
    } finally {
      active--;
    }
  };
  return { githubRequest, calls, peak: () => peak };
}

test("sync paginates all PRs, verifies merges and fetches each repository once with bounded concurrency", async t => {
  const options = await fixture(t);
  const api = githubFixture({ total: 101 });
  const snapshot = await syncContributions({ ...options, githubRequest: api.githubRequest });
  assert.equal(snapshot.pullRequests.length, 101);
  assert.equal(snapshot.repositories.length, 2);
  assert.deepEqual(contributionProjects(snapshot).map(project => project.fullName), ["org/large", "org/small"]);
  assert.deepEqual(api.calls.filter(call => call.startsWith("search/")).map(call => new URL(call, "https://api.github.com").searchParams.get("page")), ["1", "2"]);
  assert.equal(api.calls.filter(call => call.includes("/pulls/")).length, 101);
  assert.equal(api.calls.filter(call => call.startsWith("repos/") && !call.includes("/pulls/")).length, 2);
  assert.equal(api.peak(), 3);
  assert.deepEqual(JSON.parse(await readFile(options.outputPath, "utf8")), snapshot);
});

test("empty search results produce a valid empty contribution page", async t => {
  const options = await fixture(t);
  const snapshot = await syncContributions({ ...options, githubRequest: githubFixture({ total: 0 }).githubRequest });
  assert.deepEqual(snapshot.repositories, []);
  assert.deepEqual(snapshot.pullRequests, []);
});

test("partial, changing or invalid GitHub responses never replace the previous snapshot", async t => {
  const scenarios = [
    ["HTTP failure", () => { throw new Error("GitHub returned HTTP 503"); }],
    ["incomplete search", (data, url) => url.pathname === "/search/issues" ? { ...data, incomplete_results: true } : data],
    ["changed pagination", (data, url) => url.searchParams.get("page") === "2" ? { ...data, total_count: 102 } : data, 101],
    ["early pagination end", (data, url) => url.searchParams.get("page") === "2" ? { ...data, items: [] } : data, 101],
    ["missing search count", (data, url) => url.pathname === "/search/issues" ? { ...data, total_count: undefined } : data],
    ["closed unmerged PR", data => data.merged ? { ...data, merged: false } : data],
    ["wrong author", data => data.user ? { ...data, user: { login: "someone-else" } } : data],
    ["wrong PR number", data => data.merged ? { ...data, number: 2 } : data],
    ["private repository", data => "private" in data ? { ...data, private: true } : data],
    ["invalid Stars", data => "stargazers_count" in data ? { ...data, stargazers_count: -1 } : data],
    ["wrong PR link", data => data.merged ? { ...data, html_url: "https://example.test/unrelated" } : data],
    ["duplicate search result", (data, url) => url.pathname === "/search/issues" ? { ...data, items: [data.items[0], data.items[0]] } : data, 2],
  ];
  for (const [name, change, total = 1] of scenarios) {
    await t.test(name, async t => {
      const options = await fixture(t);
      await assert.rejects(syncContributions({ ...options, githubRequest: githubFixture({ total, change }).githubRequest }));
      assert.equal(await readFile(options.outputPath, "utf8"), previous);
      assert.deepEqual(await readdir(join(options.outputPath, "..")), ["contributions.json"]);
    });
  }
});

test("concurrent syncs use separate temporary files and leave one complete snapshot", async t => {
  const options = await fixture(t);
  const snapshots = await Promise.all([1, 2].map(total => syncContributions({ ...options, githubRequest: githubFixture({ total }).githubRequest })));
  const saved = JSON.parse(await readFile(options.outputPath, "utf8"));
  assert.ok(snapshots.some(snapshot => JSON.stringify(snapshot) === JSON.stringify(saved)));
  assert.deepEqual(await readdir(join(options.outputPath, "..")), ["contributions.json"]);
});
