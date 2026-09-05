// Verify a catalog entry end to end, on real hardware, in a real browser.
//
//   node scripts/verify.mjs [--entry <family>/<entry-id>]
//                           [--weights <dir>] [--runs 20] [--warmup 5]
//                           [--chunk-mb 0] [--port 8903] [--keep-open]
//                           [--out bench/verify-<stamp>.json]
//
// What it does: serves this repo plus a weights directory over localhost,
// launches Chrome with a PERSISTENT profile and the WebNN + Core ML flags,
// builds every graph the entry declares through runtime/loader.js, runs the
// family's reference prompt, checks the sha256 of the RGBA readback and the
// PSNR against both reference images, and times `runs` generations.
//
// --entry may be omitted when the catalog holds exactly one entry. It is NOT
// chosen by matching this machine: verify runs the entry you name, and reports
// what happened. If that entry was tuned for another configuration, the numbers
// will say so.
//
// --weights defaults to the workbench's IR directory, so this runs today,
// before the blobs are published anywhere.
//
// IMPORTANT: chromium.launchPersistentContext(), never chromium.launch().
// Chromium gates the Core ML backend on a non-incognito profile; with an
// ephemeral one WebNN silently falls back to TFLite/XNNPACK on the CPU, which
// is about 50x slower and reports preferredInputLayout "nhwc" instead of
// "nchw". runtime/loader.js asserts the fingerprint, so this fails loudly
// rather than quietly mis-measuring, but the launch mode is what prevents it.

import { chromium } from "playwright";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const DEFAULT_WEIGHTS = path.resolve(ROOT, "..", "webnn-workbench", "bench", "webnn", "ir");

const WEBNN_FEATURES = [
  "WebMachineLearningNeuralNetwork",
  "WebMachineLearningNeuralNetworkExperimentalFeatures",
  "WebNNCoreML",
  "WebGPUExperimentalFeatures",
];
const CHROME_ARGS = [
  "--enable-unsafe-webgpu",
  `--enable-features=${WEBNN_FEATURES.join(",")}`,
  "--enable-dawn-features=allow_unsafe_apis",
];

// ---------------------------------------------------------------------------
function parseArgs(argv) {
  const o = {
    entry: null,
    weights: DEFAULT_WEIGHTS,
    runs: 20,
    warmup: 5,
    chunkMB: 0,
    port: 8903,
    profileDir: path.join(ROOT, ".chrome-profile"),
    keepOpen: false,
    out: null,
    prompt: null,
    timeoutMs: 20 * 60 * 1000,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--entry") o.entry = next();
    else if (a === "--weights") o.weights = path.resolve(next());
    else if (a === "--runs") o.runs = parseInt(next(), 10);
    else if (a === "--warmup") o.warmup = parseInt(next(), 10);
    else if (a === "--chunk-mb") o.chunkMB = parseInt(next(), 10);
    else if (a === "--port") o.port = parseInt(next(), 10);
    else if (a === "--profile-dir") o.profileDir = path.resolve(next());
    else if (a === "--keep-open") o.keepOpen = true;
    else if (a === "--out") o.out = path.resolve(next());
    else if (a === "--prompt") o.prompt = next();
    else if (a === "--timeout-min") o.timeoutMs = parseFloat(next()) * 60000;
    else if (a === "-h" || a === "--help") { console.log(fs.readFileSync(new URL(import.meta.url), "utf8").split("\n").slice(0, 24).join("\n")); process.exit(0); }
    else throw new Error(`unknown argument ${a}`);
  }
  return o;
}

// ---------------------------------------------------------------------------
// Static server: this repo at /, the weights directory at /weights.
// Range-capable, because the chunked constant path needs 206 responses and the
// whole-file path is 1.65 GB.
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".bin": "application/octet-stream",
  ".png": "image/png",
  ".txt": "text/plain; charset=utf-8",
  ".css": "text/css; charset=utf-8",
};

function startServer({ port, mounts }) {
  const resolveUrl = (urlPath) => {
    const decoded = decodeURIComponent(urlPath.split("?")[0]);
    for (const [prefix, dir] of mounts) {
      if (decoded === prefix || decoded.startsWith(prefix + "/")) {
        const rel = decoded.slice(prefix.length).replace(/^\//, "");
        const p = path.join(dir, path.normalize("/" + rel));
        if (!p.startsWith(dir)) return null; // traversal
        return p;
      }
    }
    return null;
  };

  const server = http.createServer((req, res) => {
    res.setHeader("Cache-Control", "no-store");
    if (req.url === "/favicon.ico") return void res.writeHead(204).end();
    const filePath = resolveUrl(req.url);
    if (!filePath) return void res.writeHead(404).end(`no mount for ${req.url}`);
    let stat;
    try { stat = fs.statSync(filePath); } catch { return void res.writeHead(404).end(`not found: ${req.url}`); }
    if (!stat.isFile()) return void res.writeHead(404).end(`not a file: ${req.url}`);
    const type = MIME[path.extname(filePath).toLowerCase()] ?? "application/octet-stream";
    const range = req.headers.range && /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
    if (range) {
      const start = range[1] === "" ? stat.size - Number(range[2]) : Number(range[1]);
      const end = range[1] === "" || range[2] === "" ? stat.size - 1 : Number(range[2]);
      if (start >= 0 && end < stat.size && start <= end) {
        res.writeHead(206, {
          "Content-Type": type,
          "Content-Length": end - start + 1,
          "Content-Range": `bytes ${start}-${end}/${stat.size}`,
          "Accept-Ranges": "bytes",
        });
        return void fs.createReadStream(filePath, { start, end }).pipe(res);
      }
    }
    res.writeHead(200, { "Content-Type": type, "Content-Length": stat.size, "Accept-Ranges": "bytes" });
    fs.createReadStream(filePath).pipe(res);
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve(server)));
}

// ---------------------------------------------------------------------------
const pad = (s, n) => String(s).padEnd(n);
const rpad = (s, n) => String(s).padStart(n);

function table(rows) {
  const w = rows[0].map((_, i) => Math.max(...rows.map((r) => String(r[i]).length)));
  return rows
    .map((r, ri) =>
      "  " + r.map((c, i) => (i === 0 ? pad(c, w[i]) : rpad(c, w[i]))).join("  ") +
      (ri === 0 ? "\n  " + w.map((n) => "-".repeat(n)).join("  ") : ""),
    )
    .join("\n");
}

// ---------------------------------------------------------------------------
const args = parseArgs(process.argv.slice(2));

// ---------------------------------------------------------------------------
// Resolve the entry. The catalog index is a lookup table here, nothing more:
// an --entry that names a row is used verbatim, and a missing --entry is an
// error unless the catalog holds exactly one entry.
const catalog = JSON.parse(fs.readFileSync(path.join(ROOT, "catalog.json"), "utf8"));
const allRows = Object.entries(catalog.families).flatMap(([fam, f]) =>
  f.entries.map((e) => ({ ...e, familyId: fam, familyFile: f.family })));
if (!allRows.length) { console.error("catalog.json indexes no entries"); process.exit(2); }
let ref = args.entry;
if (!ref) {
  if (allRows.length > 1) {
    console.error(`--entry is required; the catalog holds ${allRows.length} entries:`);
    for (const r of allRows) console.error(`  ${r.familyId}/${r.id}`);
    process.exit(2);
  }
  ref = `${allRows[0].familyId}/${allRows[0].id}`;
}
const row = allRows.find((r) => `${r.familyId}/${r.id}` === ref);
if (!row) {
  console.error(`no entry "${ref}" in catalog.json. Known:`);
  for (const r of allRows) console.error(`  ${r.familyId}/${r.id}`);
  process.exit(2);
}
const entry = JSON.parse(fs.readFileSync(path.join(ROOT, row.entry), "utf8"));
const entryDir = row.path;
const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, entryDir, entry.constants), "utf8"));

for (const [key, c] of Object.entries(manifest.constants)) {
  const p = path.join(args.weights, c.file);
  if (!fs.existsSync(p)) {
    console.error(`\nmissing constants blob for "${key}": ${p}`);
    console.error(`Pass --weights <dir> pointing at a directory holding ${Object.values(manifest.constants).map((x) => x.file).join(" and ")}.`);
    console.error(`Default is the workbench IR directory: ${DEFAULT_WEIGHTS}`);
    process.exit(2);
  }
  const size = fs.statSync(p).size;
  if (size !== c.bytes) console.warn(`warning: ${c.file} is ${size} bytes, manifest says ${c.bytes}`);
}

const uptime = (() => {
  try { return execSync("uptime").toString().trim(); } catch { return "unavailable"; }
})();

console.log(`webnn-catalog verify`);
console.log(`  repo      ${ROOT}`);
console.log(`  entry     ${ref}  (variant ${entry.variant}, built for ${entry.target.backend.name} / ${entry.target.host.chip ?? entry.target.gpu.vendor} / ${entry.target.os.name} ${entry.target.os.major} / ${entry.target.browser.name} ${entry.target.browser.major})`);
console.log(`  weights   ${args.weights}`);
console.log(`  runs      ${args.runs} (warmup ${args.warmup})`);
console.log(`  constants ${args.chunkMB ? `chunked, ${args.chunkMB} MiB ranges` : "whole file"}`);
console.log(`  machine   ${os.cpus()[0]?.model ?? "?"} / ${(os.totalmem() / 2 ** 30).toFixed(0)} GB`);
console.log(`  uptime    ${uptime}`);
console.log();

const server = await startServer({
  port: args.port,
  mounts: [["/weights", args.weights], ["", ROOT]],
});
const ORIGIN = `http://localhost:${args.port}`;
console.log(`[verify] serving ${ORIGIN}`);

fs.mkdirSync(args.profileDir, { recursive: true });
const context = await chromium.launchPersistentContext(args.profileDir, {
  channel: "chrome",
  headless: false,
  args: CHROME_ARGS,
  viewport: { width: 1100, height: 900 },
});
const page = await context.newPage();
const consoleLines = [];
page.on("console", (m) => {
  const line = `[page:${m.type()}] ${m.text()}`;
  consoleLines.push(line);
  if (m.type() === "error" || m.text().startsWith("[verify]")) console.log(line);
});
page.on("pageerror", (e) => { consoleLines.push(`[pageerror] ${e.stack ?? e.message}`); console.error("[pageerror]", e.message); });

const chromeVersion = context.browser()?.version() ?? "unknown";
console.log(`[verify] chrome ${chromeVersion}`);

let result, failure = null;
try {
  await page.goto(`${ORIGIN}/scripts/verify-page.html`);
  await page.waitForFunction(() => window.__verifyReady === true, null, { timeout: 60000 });
  console.log(`[verify] building graphs (about 45 s of Core ML compilation; there is no cache)\n`);
  result = await page.evaluate(
    (o) => window.__verifyRun(o),
    { runs: args.runs, warmup: args.warmup, weightsBase: "/weights", chunkMB: args.chunkMB, prompt: args.prompt, entryPath: `/${entryDir}` },
  );
} catch (e) {
  failure = e;
  console.error(`\n[verify] FAILED: ${e.message}`);
}

// ---------------------------------------------------------------------------
if (result) {
  const c = result.checks, t = result.timings;
  const png = c.pngDataUrl;
  delete c.pngDataUrl;

  console.log(`\n=== ${ref} ===\n`);
  console.log(`backend    ${result.fingerprint.backend} (preferredInputLayout=${result.fingerprint.preferredInputLayout}, rank max ${result.fingerprint.maxRank})`);
  console.log(`chrome     ${chromeVersion}`);
  console.log(`uptime     ${uptime}\n`);

  console.log("graphs");
  console.log(table([
    ["", "ops", "consts", "constants ms", "emit ms", "build ms"],
    ["image", result.imageStats.ops, result.imageStats.constants, result.imageStats.constantsMs, result.imageStats.emitMs, result.imageStats.buildMs],
    ["text", result.textStats.ops, result.textStats.constants, result.textStats.constantsMs, result.textStats.emitMs, result.textStats.buildMs],
  ]));

  console.log("\nchecks");
  const ok = (b) => (b ? "PASS" : "FAIL");
  console.log(table([
    ["check", "value", "expected", ""],
    ["tokenizer ids", c.tokenizer.matchesReference ? "match" : "differ", "match", ok(c.tokenizer.matchesReference)],
    ["text graph maxAbs", c.textGraph.maxAbs.toExponential(3), `<= ${c.textGraph.tolerance}`, ok(c.textGraph.pass)],
    ["out sha256", c.outputSha256.slice(0, 16) + "...", c.expectedSha256 === "PENDING" ? "(pending)" : c.expectedSha256.slice(0, 16) + "...", c.expectedSha256 === "PENDING" ? "RECORD" : ok(c.sha256Match)],
    ["alpha opaque", c.alphaAllOpaque, "true", ok(c.alphaAllOpaque)],
    ["PSNR vs expected_image", c.vsExpectedImage.psnrDb, ">= 55 dB", ok(c.vsExpectedImage.psnrDb >= 55)],
    ["max byte diff vs expected", c.vsExpectedImage.maxByteDiff, "<= 2", ok(c.vsExpectedImage.maxByteDiff <= 2)],
    ["PSNR vs reference_image", c.vsReferenceImage.psnrDb, ">= 40 dB", ok(c.vsReferenceImage.psnrDb >= 40)],
    ["stable over runs", c.stableAcrossRuns, "true", ok(c.stableAcrossRuns)],
  ]));
  if (!c.vsExpectedImage.exact)
    console.log(
      `\n  Not bit-identical to expected_image.png, and it is not expected to be: that run bound one\n` +
      `  graph output, this recipe declares two (the debug \`latent\`). A different output list is a\n` +
      `  different Core ML graph. Max byte difference ${c.vsExpectedImage.maxByteDiff} on an 8-bit image is rounding.`,
    );

  console.log("\ntimings (ms)");
  console.log(table([
    ["stage", "median", "min", "max", "mean", "n", "reference"],
    ["new prompt (text + image)", t.newPrompt.medianMs, t.newPrompt.minMs, t.newPrompt.maxMs, t.newPrompt.meanMs, t.newPrompt.n, "67.85"],
    ["cached prompt (image only)", t.cachedPrompt.medianMs, t.cachedPrompt.minMs, t.cachedPrompt.maxMs, t.cachedPrompt.meanMs, t.cachedPrompt.n, "~60"],
    ["image graph + readback", t.imageGraph.medianMs, t.imageGraph.minMs, t.imageGraph.maxMs, t.imageGraph.meanMs, t.imageGraph.n, "~62"],
    ["text graph + fence", t.textGraph.medianMs, t.textGraph.minMs, t.textGraph.maxMs, t.textGraph.meanMs, t.textGraph.n, "7.15"],
  ]));

  const timingOk = Math.abs(t.newPrompt.medianMs - 67.85) <= 6;
  console.log(`\n  new-prompt median is ${(t.newPrompt.medianMs - 67.85 >= 0 ? "+" : "")}${(t.newPrompt.medianMs - 67.85).toFixed(2)} ms vs the 67.85 ms quiet-machine reference` +
    `${timingOk ? " (within noise)" : " (OUTSIDE the +-6 ms band; check machine load)"}`);

  console.log(`\nop coverage`);
  const types = Object.keys({ ...result.opCoverage.image, ...result.opCoverage.text }).sort();
  console.log(table([
    ["op", "image", "text"],
    ...types.map((k) => [k, result.opCoverage.image[k] ?? "-", result.opCoverage.text[k] ?? "-"]),
    ["TOTAL", result.imageStats.ops, result.textStats.ops],
  ]));

  if (result.errors.length) console.log(`\nERRORS:\n  ${result.errors.join("\n  ")}`);

  const failed =
    result.errors.length ||
    !c.tokenizer.matchesReference ||
    !c.textGraph.pass ||
    !c.alphaAllOpaque ||
    !c.stableAcrossRuns ||
    (c.expectedSha256 !== "PENDING" && !c.sha256Match) ||
    c.vsExpectedImage.psnrDb < 55 ||
    c.vsExpectedImage.maxByteDiff > 2 ||
    c.vsReferenceImage.psnrDb < 40;
  console.log(`\n${failed ? "VERIFY FAILED" : "VERIFY OK"}`);

  if (c.expectedSha256 === "PENDING")
    console.log(`\nRecord this in ${entryDir}/verification/expected.json:\n  "sha256": "${c.outputSha256}"`);

  const outPath = args.out ?? path.join(ROOT, "bench", `verify-${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify({
    generatedAt: new Date().toISOString(),
    chrome: chromeVersion, uptime, args: { ...args }, entry: ref, entryPath: entryDir, result, consoleLines,
  }, null, 2));
  console.log(`\nwrote ${outPath}`);
  if (png) {
    const p = outPath.replace(/\.json$/, ".png");
    fs.writeFileSync(p, Buffer.from(png.split(",")[1], "base64"));
    console.log(`wrote ${p}`);
  }
  if (failed) process.exitCode = 1;
} else {
  console.log(`\nlast page console lines:\n  ${consoleLines.slice(-25).join("\n  ")}`);
  process.exitCode = 1;
}

if (!args.keepOpen) {
  await context.close();
  server.close();
} else {
  console.log(`\n[verify] --keep-open: browser and server are still up. Ctrl-C to stop.`);
}
if (failure && !args.keepOpen) process.exitCode = 1;
