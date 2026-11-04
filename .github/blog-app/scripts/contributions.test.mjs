import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { contributionProjects, validateContributionSnapshot } from "../src/lib/contributions.ts";

const repo = (fullName, stars) => ({ fullName, name: fullName.split("/")[1], description: "", url: `https://github.com/${fullName}`, stars });
const pr = (repository, number, changes = {}) => ({
  repository, number, title: `PR ${number}`, url: `https://github.com/${repository}/pull/${number}`,
  author: "golitter", merged: true, mergedAt: "2026-11-01T00:00:00Z", ...changes,
});
const snapshot = (repositories, pullRequests) => ({ username: "golitter", syncedAt: "2026-11-04T12:00:00Z", repositories, pullRequests });

test("projects are ordered by Stars, independently of PR count or input order", () => {
  const data = snapshot([repo("org/small", 9), repo("org/large", 500), repo("org/middle", 80)], [
    pr("org/small", 1), pr("org/middle", 1), pr("org/large", 1), pr("org/small", 2),
  ]);
  assert.deepEqual(contributionProjects(data).map(project => project.fullName), ["org/large", "org/middle", "org/small"]);
});

test("only the account's explicitly merged PRs with a valid merge date are counted", () => {
  const data = snapshot([repo("org/project", 100), repo("org/no-merges", 900)], [
    pr("org/project", 1),
    pr("org/project", 2, { merged: false, mergedAt: null }),
    pr("org/project", 3, { merged: false }),
    pr("org/project", 4, { mergedAt: null }),
    pr("org/project", 5, { mergedAt: "invalid" }),
    pr("org/project", 6, { author: "someone-else" }),
    pr("org/project", 8, { merged: "false" }),
    pr("org/no-merges", 7, { merged: false, mergedAt: null }),
  ]);
  const projects = contributionProjects(data);
  assert.deepEqual(projects.map(project => project.fullName), ["org/project"]);
  assert.deepEqual(projects[0].pullRequests.map(item => item.number), [1]);
  assert.equal(projects.reduce((sum, project) => sum + project.pullRequests.length, 0), 1);
});

test("PRs stay in their project, deduplicate by project and number, and show newest merges first", () => {
  const data = snapshot([repo("org/a", 100), repo("org/b", 10)], [
    pr("org/a", 10, { mergedAt: "2026-11-01T00:00:00Z" }),
    pr("org/a", 20, { mergedAt: "2026-11-04T00:00:00Z" }),
    pr("org/a", 10), pr("org/b", 10),
  ]);
  const projects = contributionProjects(data);
  assert.deepEqual(projects[0].pullRequests.map(item => item.number), [20, 10]);
  assert.deepEqual(projects[1].pullRequests.map(item => item.number), [10]);
  assert.equal(projects.reduce((sum, project) => sum + project.pullRequests.length, 0), 3);
});

test("snapshot validation rejects malformed links, flags, fields and future dates before display", () => {
  const data = snapshot([repo("org/project", 100)], [pr("org/project", 1)]);
  const now = new Date("2026-11-04T12:00:00Z");
  assert.equal(validateContributionSnapshot(data, "GOLITTER", now), data);
  const invalidChanges = [
    value => { value.username = "another-account"; },
    value => { value.syncedAt = "2099-01-01T00:00:00Z"; },
    value => { value.syncedAt = "invalid"; },
    value => { value.repositories[0] = null; },
    value => { value.repositories[0].stars = -1; },
    value => { value.repositories[0].url = "https://example.test/unrelated"; },
    value => { value.repositories.push({ ...value.repositories[0] }); },
    value => { value.repositories.push(repo("org/unused", 1)); },
    value => { value.pullRequests[0] = null; },
    value => { value.pullRequests[0].merged = "false"; },
    value => { value.pullRequests[0].mergedAt = null; },
    value => { value.pullRequests[0].mergedAt = "2027-01-01T00:00:00Z"; },
    value => { value.pullRequests[0].title = ""; },
    value => { value.pullRequests[0].number = 0; },
    value => { value.pullRequests[0].author = "another-account"; },
    value => { value.pullRequests[0].repository = "org/missing"; },
    value => { value.pullRequests[0].url = "https://github.com/org/project/pull/2"; },
    value => { value.pullRequests.push({ ...value.pullRequests[0] }); },
  ];
  for (const change of invalidChanges) {
    const invalid = structuredClone(data);
    change(invalid);
    assert.throws(() => validateContributionSnapshot(invalid, "golitter", now));
  }
});

test("the real snapshot is complete and contains only verified public GitHub PR links", async () => {
  const data = JSON.parse(await readFile(new URL("../src/generated/contributions.json", import.meta.url), "utf8"));
  const projects = contributionProjects(data);
  assert.equal(projects.reduce((sum, project) => sum + project.pullRequests.length, 0), data.pullRequests.length);
  assert.equal(projects.length, data.repositories.length);
  for (const project of projects) {
    assert.ok(Number.isInteger(project.stars) && project.stars >= 0);
    for (const item of project.pullRequests) {
      assert.equal(item.url, `https://github.com/${project.fullName}/pull/${item.number}`);
      assert.equal(item.merged, true);
    }
  }
});
