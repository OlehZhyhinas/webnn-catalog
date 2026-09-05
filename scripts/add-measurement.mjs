// Append a measurements row to an existing entry.
//
//   node scripts/add-measurement.mjs --entry <family>/<entry-id>
//                                    --results <file> --target <target.json>
//                                    [--protocol e2e|replay|interleaved-ab|isolated-block]
//                                    [--label ...] [--notes ...] [--date ISO]
//                                    [--source <what the numbers came from>] [--replace]
//
// A different machine is a different ROW, not a different entry: the graphs are
// byte-identical, so what changes is only what they cost. Every row carries its
// own host fingerprint, taken from a target.json produced on the machine that
// ran it.
//
// Recognised `--results` shapes:
//   * webnn-workbench e2e harness JSON  (result.timings.totalNewPrompt)
//   * this repo's scripts/verify.mjs    (result.timings.newPrompt)
//   * a plain object                    ({newPromptMs, cachedMs, stages, runs})

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const readJson = (p) => JSON.parse(fs.readFileSync(p, "utf8"));

/** The subset of a target that identifies the machine a row was taken on. */
export function hostFromTarget(target) {
  return {
    backend: target.backend.name,
    deviceType: target.backend.deviceType,
    browser: `${target.browser.name} ${target.browser.version}`,
    os: `${target.os.name} ${target.os.version}${target.os.build ? ` (${target.os.build})` : ""}`,
    chip: target.host?.chip ?? null,
    gpu: [target.gpu?.vendor, target.gpu?.architecture].filter(Boolean).join("/") || null,
    gpuCores: target.host?.gpuCores ?? null,
    memoryGB: target.host?.memoryGB ?? null,
    f16TflopsMeasured: target.perfClass?.f16TflopsMeasured ?? null,
  };
}

const num = (v) => (typeof v === "number" ? +v.toFixed(3) : null);

/** Turn a results file into a row. Shape-detecting, and it invents nothing. */
export function rowFromResults(results, target, opts = {}) {
  const host = { ...hostFromTarget(target), sameAsTarget: opts.sameAsTarget ?? true };
  const row = {
    ...(opts.label ? { label: opts.label } : {}),
    host,
    date: opts.date ?? new Date().toISOString().slice(0, 10),
    protocol: opts.protocol ?? "e2e",
  };

  const t = results?.result?.timings;
  if (t?.totalNewPrompt) {
    // webnn-workbench e2e harness
    const s = t.staged ?? {};
    row.runs = t.totalNewPrompt.n ?? null;
    row.warmup = results.harness?.args?.warmup ?? results.result?.opts?.warmup ?? null;
    row.statistic = "median of the timed runs";
    row.newPromptMs = num(t.totalNewPrompt.medianMs);
    row.cachedMs = num(t.totalCachedEmbedding?.medianMs);
    row.stages = {
      tokenize: num(t.tokenizeNewPrompt?.medianMs),
      textGraph: num(s.textEncoderPlusFence?.medianMs),
      imageGraph: num(s.imageGraphPlusReadback?.medianMs),
      readback: num(s.readback?.medianMs),
      toImageData: num(s.toImageData?.medianMs),
      putImageData: num(s.putImageData?.medianMs),
    };
    row.buildOnceMs = { image: num(t.imageBuildMs), text: num(t.textBuildMs), firstDispatch: num(t.firstDispatchMs) };
    const q = results.result?.verification?.image;
    if (q) row.quality = { psnr8VsFp32Cpu: num(q.psnr8VsFp32Cpu), psnr8VsFp16Mps: num(q.psnr8VsFp16Mps) };
  } else if (t?.newPrompt) {
    // this repo's scripts/verify.mjs
    row.runs = t.newPrompt.n ?? null;
    row.warmup = results.args?.warmup ?? null;
    row.statistic = "median of the timed runs";
    row.newPromptMs = num(t.newPrompt.medianMs);
    row.cachedMs = num(t.cachedPrompt?.medianMs);
    row.stages = {
      textGraph: num(t.textGraph?.medianMs),
      imageGraph: num(t.imageGraph?.medianMs),
    };
    row.buildOnceMs = {
      image: num(results.result?.imageStats?.buildMs),
      text: num(results.result?.textStats?.buildMs),
      imageConstants: num(results.result?.imageStats?.constantsMs),
      imageEmit: num(results.result?.imageStats?.emitMs),
    };
    row.outputSha256 = results.result?.checks?.outputSha256 ?? null;
    const c = results.result?.checks;
    if (c) row.quality = {
      psnrVsExpectedImageDb: c.vsExpectedImage?.psnrDb ?? null,
      psnrVsReferenceImageDb: c.vsReferenceImage?.psnrDb ?? null,
      textEncoderMaxAbs: c.textGraph?.maxAbs ?? null,
    };
  } else if (typeof results?.newPromptMs === "number" || typeof results?.newPrompt === "number") {
    row.newPromptMs = num(results.newPromptMs ?? results.newPrompt);
    row.cachedMs = num(results.cachedMs ?? results.cached ?? null);
    if (results.stages) row.stages = results.stages;
    if (results.runs) row.runs = results.runs;
    if (results.statistic) row.statistic = results.statistic;
    if (results.buildOnceMs) row.buildOnceMs = results.buildOnceMs;
    if (results.quality) row.quality = results.quality;
    if (results.outputSha256) row.outputSha256 = results.outputSha256;
  } else {
    throw new Error("unrecognised --results shape: expected a workbench e2e JSON, a verify.mjs JSON, or {newPromptMs, cachedMs, stages}");
  }

  // Load average, only if the run itself recorded one or the caller passes it.
  // A busy machine is a slow machine, so an absolute reading without its load
  // is misleading; but the target's load is the load at PROBE time, which is a
  // different moment, so it is not a substitute and is never borrowed here.
  const load = results.loadAvg ?? results.result?.loadAvg ?? null;
  if (load !== null) row.loadAvg = load;
  if (opts.loadAvg !== undefined && opts.loadAvg !== null) row.loadAvg = opts.loadAvg;
  if (opts.notes) row.notes = opts.notes;
  if (opts.source) row.source = opts.source;

  for (const k of Object.keys(row)) if (row[k] === null || row[k] === undefined) delete row[k];
  if (row.stages) for (const k of Object.keys(row.stages)) if (row.stages[k] === null) delete row.stages[k];
  if (row.buildOnceMs) for (const k of Object.keys(row.buildOnceMs)) if (row.buildOnceMs[k] === null) delete row.buildOnceMs[k];
  return row;
}

/** Append (or replace by label) a row in an entry's measurements.json. */
export function appendRow(family, entryId, row, { replace = false } = {}) {
  const entryDir = path.join(ROOT, "families", family, "entries", entryId);
  const entry = readJson(path.join(entryDir, "entry.json"));
  const mPath = path.join(entryDir, entry.measurements ?? "measurements.json");
  const m = fs.existsSync(mPath) ? readJson(mPath) : { entry: entryId, family, rows: [] };
  const at = row.label ? m.rows.findIndex((r) => r.label === row.label) : -1;
  if (at >= 0 && !replace) throw new Error(`a row labelled "${row.label}" already exists; pass --replace to overwrite it`);
  if (at >= 0) m.rows[at] = row; else m.rows.push(row);
  fs.writeFileSync(mPath, JSON.stringify(m, null, 2) + "\n");
  return { path: mPath, rows: m.rows.length };
}

/** Keep the catalog index's row count honest after a measurements change. */
export function refreshCatalogMeasuredHosts(family, entryId) {
  const catalogPath = path.join(ROOT, "catalog.json");
  const catalog = readJson(catalogPath);
  const entryDir = path.join(ROOT, "families", family, "entries", entryId);
  const entry = readJson(path.join(entryDir, "entry.json"));
  const mPath = path.join(entryDir, entry.measurements ?? "measurements.json");
  if (!fs.existsSync(mPath)) return;
  const m = readJson(mPath);
  const row = catalog.families[family]?.entries.find((e) => e.id === entryId);
  if (!row) return;
  row.measuredHosts = new Set(m.rows.map((r) => r.host.chip ?? r.host.browser)).size;
  row.newPromptMs = m.rows[0]?.newPromptMs ?? null;
  fs.writeFileSync(catalogPath, JSON.stringify(catalog, null, 2) + "\n");
}

// ---------------------------------------------------------------------------
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const o = { protocol: "e2e", replace: false };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i], next = () => argv[++i];
    if (a === "--entry") o.entry = next();
    else if (a === "--results") o.results = path.resolve(next());
    else if (a === "--target") o.target = path.resolve(next());
    else if (a === "--protocol") o.protocol = next();
    else if (a === "--label") o.label = next();
    else if (a === "--notes") o.notes = next();
    else if (a === "--date") o.date = next();
    else if (a === "--source") o.source = next();
    else if (a === "--load-avg") o.loadAvg = parseFloat(next());
    else if (a === "--replace") o.replace = true;
    else if (a === "-h" || a === "--help") {
      console.log(fs.readFileSync(new URL(import.meta.url), "utf8").split("\n").slice(0, 19).join("\n"));
      process.exit(0);
    } else throw new Error(`unknown argument ${a}`);
  }
  for (const need of ["entry", "results", "target"]) if (!o[need]) throw new Error(`--${need} is required`);
  const [family, entryId] = o.entry.split("/");
  if (!family || !entryId) throw new Error(`--entry must be <family>/<entry-id>`);

  const target = readJson(o.target);
  const row = rowFromResults(readJson(o.results), target, o);
  const { path: p, rows } = appendRow(family, entryId, row, { replace: o.replace });
  refreshCatalogMeasuredHosts(family, entryId);
  console.log(JSON.stringify(row, null, 2));
  console.log(`\nappended to ${path.relative(ROOT, p)} (${rows} rows)`);
}
