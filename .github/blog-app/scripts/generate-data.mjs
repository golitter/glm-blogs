import { spawnSync } from "node:child_process";

// Windows commonly exposes `python`; GitHub Actions exposes `python3`.
const candidates = process.platform === "win32" ? ["python", "python3"] : ["python3", "python"];
for (const executable of candidates) {
  const result = spawnSync(executable, ["-X", "utf8", "scripts/generate-blog-data.py"], { stdio: "inherit" });
  if (result.error?.code === "ENOENT") continue;
  if (result.error) console.error(result.error.message);
  process.exit(result.status ?? 1);
}
console.error("Python 3 is required to generate the blog data. Install Python and add it to PATH.");
process.exit(1);
