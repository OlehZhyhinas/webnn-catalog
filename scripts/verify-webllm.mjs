import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { withGpuLock } from "./gpu-lock.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const DEFAULT_ENTRY =
  "https://raw.githubusercontent.com/OlehZhyhinas/webnn-catalog/main/" +
  "families/qwen3-0.6b-q4f16-1/entries/" +
  "webgpu-apple-m5-pro-macos26-chrome152-sg32-burst4/entry.json";

const options = {
  entryUrl: DEFAULT_ENTRY,
  profile: path.join(ROOT, "bench/.chrome-profile-qwen"),
  output: null,
  cases: null,
  timeoutMs: 20 * 60 * 1000,
  port: 8906,
};
const argv = process.argv.slice(2);
for (let index = 0; index < argv.length; index++) {
  const value = argv[index];
  if (value === "--entry-url") options.entryUrl = argv[++index];
  else if (value === "--profile") options.profile = path.resolve(argv[++index]);
  else if (value === "--output") options.output = path.resolve(argv[++index]);
  else if (value === "--cases") options.cases = Number(argv[++index]);
  else if (value === "--timeout-ms") options.timeoutMs = Number(argv[++index]);
  else if (value === "--port") options.port = Number(argv[++index]);
  else throw new Error(`unknown argument: ${value}`);
}

function startServer() {
  const child = spawn(
    process.execPath,
    [path.join(HERE, "serve.mjs"), "--port", String(options.port)],
    { cwd: ROOT, stdio: ["ignore", "pipe", "inherit"] },
  );
  const ready = new Promise((resolve, reject) => {
    let text = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      process.stdout.write(chunk);
      text += chunk;
      if (text.includes(`http://localhost:${options.port}/`)) resolve();
    });
    child.once("exit", (code) =>
      reject(new Error(`catalog server exited before readiness (${code})`)),
    );
    child.once("error", reject);
  });
  return { child, ready };
}

await withGpuLock("catalog Qwen public verification", async (loadAvg) => {
  const server = startServer();
  let context;
  try {
    await server.ready;
    await mkdir(options.profile, { recursive: true });
    context = await chromium.launchPersistentContext(options.profile, {
      channel: "chrome",
      headless: false,
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=WebGPUExperimentalFeatures",
        "--enable-dawn-features=allow_unsafe_apis",
        "--no-sandbox",
      ],
      viewport: { width: 1200, height: 800 },
    });
    const page = context.pages()[0] ?? (await context.newPage());
    page.setDefaultTimeout(options.timeoutMs);
    page.setDefaultNavigationTimeout(options.timeoutMs);
    page.on("console", (message) => {
      if (message.type() !== "debug") {
        console.log(`[page:${message.type()}] ${message.text()}`);
      }
    });
    page.on("pageerror", (error) =>
      console.error(`[pageerror] ${error.message}`),
    );
    await page.goto(
      `http://localhost:${options.port}/verification/webllm.html`,
      { waitUntil: "domcontentloaded" },
    );
    await page.waitForFunction(() => typeof globalThis.__verifyWebLLM === "function");
    const result = await page.evaluate(
      ({ entryUrl, cases }) =>
        globalThis.__verifyWebLLM(entryUrl, { maxCases: cases }),
      { entryUrl: options.entryUrl, cases: options.cases },
    );
    const report = {
      ...result,
      entryUrl: options.entryUrl,
      verifiedAt: new Date().toISOString(),
      chrome: await page.evaluate(() => navigator.userAgent),
      loadAvg,
    };
    console.log(JSON.stringify(report, null, 2));
    if (options.output) {
      await writeFile(options.output, `${JSON.stringify(report, null, 2)}\n`);
    }
  } finally {
    await context?.close();
    server.child.kill("SIGTERM");
  }
}, { maxLoad: process.env.WEBNN_MAX_LOAD === undefined ? 4 : Number(process.env.WEBNN_MAX_LOAD) });
