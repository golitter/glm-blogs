import { appendFile, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { PROFILE_URL } from "../src/lib/constants.ts";
import { validateContributionSnapshot } from "../src/lib/contributions.ts";

export const CONTRIBUTION_SYNC_DAYS = 5;
const DAY_MS = 86_400_000;
const CHINA_OFFSET_MS = 8 * 60 * 60 * 1000;
const username = new URL(PROFILE_URL).pathname.split("/").filter(Boolean)[0];

export function contributionSyncDue(syncedAt, now = new Date()) {
  const previous = Date.parse(syncedAt);
  if (!Number.isFinite(previous) || !Number.isFinite(now.getTime())) throw new Error("Invalid contribution sync date");
  const chinaDay = timestamp => Math.floor((timestamp + CHINA_OFFSET_MS) / DAY_MS);
  return chinaDay(now.getTime()) - chinaDay(previous) >= CONTRIBUTION_SYNC_DAYS;
}

export async function prepareContributions({
  eventName = process.env.GITHUB_EVENT_NAME ?? "",
  snapshotUrl = process.env.CONTRIBUTIONS_SNAPSHOT_URL,
  snapshotPath = new URL("../src/generated/contributions.json", import.meta.url),
  outputPath = process.env.GITHUB_OUTPUT,
  now = new Date(),
  fetchImpl = fetch,
} = {}) {
  const local = validateContributionSnapshot(JSON.parse(await readFile(snapshotPath, "utf8")), username, now);
  let published;

  // PR checks stay offline. A manual run always requests a fresh GitHub sync.
  if (snapshotUrl && eventName !== "pull_request" && eventName !== "workflow_dispatch") {
    const response = await fetchImpl(snapshotUrl, {
      headers: { Accept: "application/json", "Cache-Control": "no-cache" },
      cache: "no-store", signal: AbortSignal.timeout(20_000),
    });
    if (response.ok) published = validateContributionSnapshot(await response.json(), username, now);
    else if (response.status !== 404) throw new Error(`Cannot restore published contributions: HTTP ${response.status}`);
  }

  // The deployed snapshot is the last successfully published update. A failed deployment
  // cannot move this date forward, so the next daily check retries overdue updates.
  const lastPublished = published ?? local;
  const due = contributionSyncDue(lastPublished.syncedAt, now);
  const shouldSync = eventName === "workflow_dispatch" || (eventName !== "pull_request" && due);
  const shouldBuild = eventName !== "schedule" || shouldSync;

  // Ordinary blog pushes must keep newer scheduled data instead of rolling back to
  // the older checked-in snapshot. A newer manually synced local snapshot is retained.
  if (published && Date.parse(published.syncedAt) > Date.parse(local.syncedAt)) {
    await writeFile(snapshotPath, JSON.stringify(published, null, 2) + "\n", "utf8");
  }
  if (outputPath) {
    await appendFile(outputPath, `should_build=${shouldBuild}\nshould_sync=${shouldSync}\n`, "utf8");
  }
  return { shouldBuild, shouldSync };
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  prepareContributions().then(({ shouldBuild, shouldSync }) => {
    console.log(`Contribution refresh: ${shouldSync ? "due" : "not due"}; page build: ${shouldBuild ? "yes" : "skipped"}. Interval: ${CONTRIBUTION_SYNC_DAYS} China calendar days.`);
  }).catch(error => {
    console.error(`Contribution preparation failed: ${error.message}`);
    process.exitCode = 1;
  });
}
