import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { contributionSyncDue, prepareContributions } from "./prepare-contributions.mjs";

// Fixed fixtures keep historical scheduling tests independent of future real PR merges.
const seed = {
  username: "golitter",
  repositories: [{ fullName: "org/project", name: "project", description: "", url: "https://github.com/org/project", stars: 100 }],
  pullRequests: [{ repository: "org/project", number: 1, title: "Fix a bug", url: "https://github.com/org/project/pull/1", author: "golitter", merged: true, mergedAt: "2026-11-01T00:00:00Z" }],
};
const snapshot = syncedAt => ({ ...structuredClone(seed), syncedAt });
async function fixture(t, syncedAt = "2026-11-04T12:04:00Z") {
  const directory = await mkdtemp(path.join(tmpdir(), "glm-contribution-schedule-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const snapshotPath = path.join(directory, "contributions.json");
  const outputPath = path.join(directory, "outputs.txt");
  await writeFile(snapshotPath, JSON.stringify(snapshot(syncedAt)));
  return { snapshotPath, outputPath, snapshotUrl: "https://example.test/blog/contributions.json", now: new Date("2026-11-09T01:17:00Z") };
}
const response = (syncedAt, status = 200) => new Response(status === 200 ? JSON.stringify(snapshot(syncedAt)) : null, { status });

test("refresh is due on the fifth China calendar day, including midnight boundaries", () => {
  const previous = "2026-11-04T12:04:00Z";
  assert.equal(contributionSyncDue(previous, new Date("2026-11-08T01:17:00Z")), false);
  assert.equal(contributionSyncDue(previous, new Date("2026-11-08T15:59:59Z")), false);
  assert.equal(contributionSyncDue(previous, new Date("2026-11-08T16:00:00Z")), true);
  assert.equal(contributionSyncDue(previous, new Date("2026-11-09T01:17:00Z")), true);
});

test("five-day intervals survive month ends, year ends and leap years", () => {
  for (const [previous, due] of [
    ["2026-11-29", "2026-12-04"], ["2026-12-29", "2027-01-03"],
    ["2027-01-31", "2027-02-05"], ["2028-02-27", "2028-03-03"],
  ]) {
    const deadline = new Date(`${due}T01:17:00Z`);
    assert.equal(contributionSyncDue(`${previous}T01:17:00Z`, new Date(+deadline - 86_400_000)), false);
    assert.equal(contributionSyncDue(`${previous}T01:17:00Z`, deadline), true);
  }
});

test("scheduled checks skip publication until five days after the published update", async t => {
  const options = await fixture(t);
  const result = await prepareContributions({ ...options, eventName: "schedule", fetchImpl: async () => response("2026-11-08T01:17:00Z") });
  assert.deepEqual(result, { shouldBuild: false, shouldSync: false });
  assert.equal(JSON.parse(await readFile(options.snapshotPath, "utf8")).syncedAt, "2026-11-08T01:17:00Z");
  assert.equal(await readFile(options.outputPath, "utf8"), "should_build=false\nshould_sync=false\n");
});

test("due updates retry until a newly published snapshot advances the interval", async t => {
  const options = await fixture(t, "2026-11-08T12:00:00Z");
  const result = await prepareContributions({ ...options, eventName: "schedule", fetchImpl: async () => response("2026-11-04T12:04:00Z") });
  assert.deepEqual(result, { shouldBuild: true, shouldSync: true });
});

test("normal blog builds retain newer published PR data without refreshing too early", async t => {
  const options = await fixture(t);
  const result = await prepareContributions({ ...options, eventName: "push", fetchImpl: async () => response("2026-11-08T01:17:00Z") });
  assert.deepEqual(result, { shouldBuild: true, shouldSync: false });
  assert.equal(JSON.parse(await readFile(options.snapshotPath, "utf8")).syncedAt, "2026-11-08T01:17:00Z");
});

test("PR checks stay offline and manual runs force a refresh", async t => {
  const options = await fixture(t, "2026-11-09T01:00:00Z");
  const fetchImpl = async () => { throw new Error("Network must not be used for these events"); };
  assert.deepEqual(await prepareContributions({ ...options, eventName: "pull_request", fetchImpl }), { shouldBuild: true, shouldSync: false });
  assert.deepEqual(await prepareContributions({ ...options, eventName: "workflow_dispatch", fetchImpl }), { shouldBuild: true, shouldSync: true });
});

test("the first deployment can fall back to the checked-in snapshot when no public snapshot exists", async t => {
  const options = await fixture(t);
  assert.deepEqual(await prepareContributions({ ...options, eventName: "schedule", fetchImpl: async () => response(null, 404) }), { shouldBuild: true, shouldSync: true });
});

test("failed or invalid snapshot responses preserve the previous data and fail the build", async t => {
  const options = await fixture(t);
  const original = await readFile(options.snapshotPath, "utf8");
  for (const fetchImpl of [
    async () => response(null, 503),
    async () => new Response(JSON.stringify({ ...snapshot("2026-11-08T01:17:00Z"), username: "someone-else" })),
    async () => new Response(JSON.stringify({ ...snapshot("2026-11-08T01:17:00Z"), syncedAt: "invalid" })),
    async () => new Response(JSON.stringify(snapshot("2099-01-01T00:00:00Z"))),
    async () => {
      const invalid = snapshot("2026-11-08T01:17:00Z");
      invalid.pullRequests[0].merged = "false";
      return new Response(JSON.stringify(invalid));
    },
    async () => {
      const invalid = snapshot("2026-11-08T01:17:00Z");
      invalid.pullRequests[0].url = "https://example.test/unrelated";
      return new Response(JSON.stringify(invalid));
    },
  ]) {
    await assert.rejects(prepareContributions({ ...options, eventName: "schedule", fetchImpl }));
    assert.equal(await readFile(options.snapshotPath, "utf8"), original);
    await assert.rejects(readFile(options.outputPath, "utf8"), { code: "ENOENT" });
  }
});

test("the built public snapshot matches the data shown on the page", async () => {
  const checkedIn = JSON.parse(await readFile(new URL("../src/generated/contributions.json", import.meta.url), "utf8"));
  const published = JSON.parse(await readFile(new URL("../public/contributions.json", import.meta.url), "utf8"));
  assert.deepEqual(published, checkedIn);
});
