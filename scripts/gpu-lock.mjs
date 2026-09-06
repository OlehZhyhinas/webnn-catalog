import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";

export const LOCK_DIR = "/tmp/webnn-workbench-gpu.lock";
const OWNER = path.join(LOCK_DIR, "owner.json");
const MAX_LOAD =
  process.env.WEBNN_MAX_LOAD === undefined
    ? 4
    : Number(process.env.WEBNN_MAX_LOAD);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function pidAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === "EPERM";
  }
}

function owner() {
  try {
    return JSON.parse(readFileSync(OWNER, "utf8"));
  } catch {
    return null;
  }
}

export async function acquireGpuLock(
  label,
  { timeoutMs = 60 * 60 * 1000, pollMs = 5000 } = {},
) {
  const started = Date.now();
  for (;;) {
    try {
      mkdirSync(LOCK_DIR);
      writeFileSync(
        OWNER,
        JSON.stringify({
          pid: process.pid,
          label,
          since: new Date().toISOString(),
          cwd: process.cwd(),
        }),
      );
      return;
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
    }
    const current = owner();
    if (current && !pidAlive(current.pid)) {
      rmSync(LOCK_DIR, { recursive: true, force: true });
      continue;
    }
    if (Date.now() - started > timeoutMs) {
      throw new Error(`timed out waiting for ${LOCK_DIR}`);
    }
    await sleep(pollMs);
  }
}

async function waitForQuiet({
  maxLoad = MAX_LOAD,
  timeoutMs = 30 * 60 * 1000,
  pollMs = 15000,
} = {}) {
  if (!maxLoad) return os.loadavg();
  const started = Date.now();
  while (os.loadavg()[0] > maxLoad && Date.now() - started <= timeoutMs) {
    await sleep(pollMs);
  }
  return os.loadavg();
}

export function releaseGpuLock() {
  const current = owner();
  if (current && current.pid !== process.pid) return;
  if (existsSync(LOCK_DIR)) rmSync(LOCK_DIR, { recursive: true, force: true });
}

export async function withGpuLock(label, fn, options = {}) {
  await acquireGpuLock(label, options);
  const release = () => releaseGpuLock();
  process.on("exit", release);
  try {
    const loadAvg = await waitForQuiet(options);
    return await fn(loadAvg);
  } finally {
    release();
    process.off("exit", release);
  }
}
