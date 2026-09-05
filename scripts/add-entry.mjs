// Create a catalog entry from the files a tuning run produced.
//
//   node scripts/add-entry.mjs
//     --family sd-turbo-512-1step
//     [--entry-id <id>]                   default: derived from the target
//     --target target.json                from scripts/probe-target.mjs
//     --recipe image=<path> --recipe text=<path>          (repeatable)
//     --constants image=<path> --constants text=<path>    (repeatable)
//     [--chain text.out=image.encoder_hidden_states]      (repeatable)
//     [--variant exact|<name>]            default: exact
//     [--measurements <results.json>] [--measurement-label ...] [--protocol e2e]
//     [--verification <dir>] [--provenance <file.md>]
//     [--produced-by "<repo>@<commit>"] [--recorder "..."]
//     [--title ...] [--summary ...] [--role image="UNet + ..."]
//     [--requires backend.name=coreml]    (repeatable; default: backend.name)
//     [--force]
//
// It reads the recipes to fill in the entry's I/O contract, hashes the recipes
// and the constants blobs (without copying the blobs: they are 1.65 GB and they
// never enter git), writes entry.json + manifest.json + measurements.json,
// copies the verification set and the provenance ledger in, adds a summary row
// to catalog.json, and then validates everything it wrote.
//
// What it does NOT do: decide anything. An entry records the configuration it
// was built for. Which entry a given user should be served is the consuming
// product's problem, and nothing in this repo has an opinion about it.

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { rowFromResults } from "./add-measurement.mjs";
import { validateCatalog, createValidator, checkRecipe } from "./validate.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const readJson = (p) => JSON.parse(fs.readFileSync(p, "utf8"));
const writeJson = (p, o) => fs.writeFileSync(p, JSON.stringify(o, null, 2) + "\n");
// Copying a file onto itself truncates it. That happens whenever an entry is
// rebuilt in place with --force from its own files, which is exactly what a
// re-run after a schema change looks like.
const copyIfDifferent = (src, dest) => {
  if (path.resolve(src) === path.resolve(dest)) return false;
  fs.copyFileSync(src, dest);
  return true;
};
const copyDirIfDifferent = (src, dest) => {
  if (path.resolve(src) === path.resolve(dest)) return false;
  fs.cpSync(src, dest, { recursive: true });
  return true;
};
const sha256File = (p) => {
  const h = crypto.createHash("sha256");
  const fd = fs.openSync(p, "r");
  const buf = Buffer.alloc(1 << 22);
  let n;
  while ((n = fs.readSync(fd, buf, 0, buf.length, null)) > 0) h.update(buf.subarray(0, n));
  fs.closeSync(fd);
  return h.digest("hex");
};
const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

// ---------------------------------------------------------------------------
const o = { recipes: {}, constants: {}, chain: [], roles: {}, requires: {}, variant: "exact", protocol: "e2e", force: false };
const argv = process.argv.slice(2);
const pair = (s) => {
  const i = s.indexOf("=");
  if (i < 0) throw new Error(`expected name=value, got "${s}"`);
  return [s.slice(0, i), s.slice(i + 1)];
};
for (let i = 0; i < argv.length; i++) {
  const a = argv[i], next = () => argv[++i];
  if (a === "--family") o.family = next();
  else if (a === "--entry-id") o.entryId = next();
  else if (a === "--target") o.target = path.resolve(next());
  else if (a === "--recipe") { const [k, v] = pair(next()); o.recipes[k] = path.resolve(v); }
  else if (a === "--recorded-from") { const [k, v] = pair(next()); (o.recordedFrom ??= {})[k] = v; }
  else if (a === "--measurement-load-avg") o.measurementLoadAvg = parseFloat(next());
  else if (a === "--constants") { const [k, v] = pair(next()); o.constants[k] = path.resolve(v); }
  else if (a === "--chain") { const [k, v] = pair(next()); o.chain.push([k, v]); }
  else if (a === "--role") { const [k, v] = pair(next()); o.roles[k] = v; }
  else if (a === "--readable") { const [k, v] = pair(next()); (o.readable ??= {})[k] = v.split(",").filter(Boolean); }
  else if (a === "--output-role") { const [k, v] = pair(next()); (o.outputRoles ??= {})[k] = v; }
  else if (a === "--measurements-extra") o.measurementsExtra = path.resolve(next());
  else if (a === "--requires") { const [k, v] = pair(next()); o.requires[k] = v; }
  else if (a === "--variant") o.variant = next();
  else if (a === "--measurements") o.measurements = path.resolve(next());
  else if (a === "--measurement-label") o.measurementLabel = next();
  else if (a === "--measurement-notes") o.measurementNotes = next();
  else if (a === "--measurement-date") o.measurementDate = next();
  else if (a === "--measurement-source") o.measurementSource = next();
  else if (a === "--protocol") o.protocol = next();
  else if (a === "--verification") o.verification = path.resolve(next());
  else if (a === "--provenance") o.provenance = path.resolve(next());
  else if (a === "--produced-by") o.producedBy = next();
  else if (a === "--recorder") o.recorder = next();
  else if (a === "--title") o.title = next();
  else if (a === "--summary") o.summary = next();
  else if (a === "--force") o.force = true;
  else if (a === "-h" || a === "--help") {
    console.log(fs.readFileSync(new URL(import.meta.url), "utf8").split("\n").slice(0, 26).join("\n"));
    process.exit(0);
  } else throw new Error(`unknown argument ${a}`);
}
if (!o.family) throw new Error("--family is required");
if (!o.target) throw new Error("--target is required (run scripts/probe-target.mjs first)");
if (!Object.keys(o.recipes).length) throw new Error("at least one --recipe name=path is required");

// ---------------------------------------------------------------------------
const v = createValidator();
const target = readJson(o.target);
{
  const errs = v.validate("target.schema.json", target, path.basename(o.target));
  if (errs.length) { console.error(`${o.target} is not a valid target:\n  ${errs.join("\n  ")}`); process.exit(2); }
}

const famDir = path.join(ROOT, "families", o.family);
const famFile = path.join(famDir, "family.json");
if (!fs.existsSync(famFile))
  throw new Error(`no ${path.relative(ROOT, famFile)}. A family is authored by hand: it is the contract every entry must meet.`);
const family = readJson(famFile);
{
  const errs = v.validate("family.schema.json", family, path.relative(ROOT, famFile));
  if (errs.length) { console.error(`${path.relative(ROOT, famFile)} is not valid:\n  ${errs.join("\n  ")}`); process.exit(2); }
}

// The entry id is a readable summary of the target, not a key anything parses.
const entryId = o.entryId ?? [
  target.backend.name,
  slug(target.host?.chip || `${target.gpu.vendor}-${target.gpu.architecture}`),
  `${slug(target.os.name)}${target.os.major}`,
  `${slug(target.browser.name)}${target.browser.major}`,
].filter(Boolean).join("-");

const entryDir = path.join(famDir, "entries", entryId);
if (fs.existsSync(entryDir) && !o.force)
  throw new Error(`${path.relative(ROOT, entryDir)} already exists. Pass --force to overwrite it, or pick another --entry-id.`);
fs.mkdirSync(entryDir, { recursive: true });

// ---------------------------------------------------------------------------
// Recipes: read, check, copy in.
const graphs = {};
const recipeHashes = {};
const allOps = new Set();
for (const [name, src] of Object.entries(o.recipes)) {
  const recipe = readJson(src);
  const label = `${name} recipe (${path.basename(src)})`;
  const errs = [...v.validate("recipe.schema.json", recipe, label), ...checkRecipe(recipe, label)];
  if (errs.length) { console.error(`${src} is not a valid recipe:\n  ${errs.join("\n  ")}`); process.exit(2); }

  const file = `recipe.${name}.json`;
  copyIfDifferent(src, path.join(entryDir, file));
  // `recordedFrom` names the dump this file came out of, and `sourceSha256`
  // pins it, so a recipe that has been touched since is detectable.
  const origin = o.recordedFrom?.[name] ?? src;
  const originPath = path.resolve(origin);
  recipeHashes[file] = {
    sha256: sha256File(path.join(entryDir, file)),
    recordedFrom: fs.existsSync(originPath) ? path.relative(path.dirname(ROOT), originPath) : origin,
    ...(fs.existsSync(originPath) && sha256File(originPath) !== sha256File(path.join(entryDir, file))
      ? { sourceSha256: sha256File(originPath) }
      : {}),
  };

  const histo = {};
  for (const op of recipe.ops) histo[op.type] = (histo[op.type] ?? 0) + 1;
  for (const t of Object.keys(histo)) allOps.add(t);
  if (!o.constants[name]) throw new Error(`graph "${name}" has no --constants ${name}=<blob>`);

  graphs[name] = {
    recipe: file,
    constants: name,
    ...(o.roles[name] ?? recipe.provenance?.summary ? { role: o.roles[name] ?? recipe.provenance.summary } : {}),
    ...(recipe.layout ? { layout: recipe.layout } : {}),
    ops: recipe.ops.length,
    opTypes: Object.keys(histo).length,
    constantCount: Object.keys(recipe.constants).length,
    inputs: recipe.inputs.map((i) => ({ name: i.name, dataType: i.dataType, shape: i.shape })),
    // `readable` says which outputs a caller is expected to read back. WebNN
    // makes it bind ALL of them regardless, so a debug output the recording
    // carried along is still allocated; it is just allocated non-readable.
    outputs: recipe.outputs.map((x) => ({
      name: x.name,
      dataType: x.dataType,
      shape: x.shape,
      ...(o.outputRoles?.[`${name}.${x.name}`] ?? x.role ? { role: o.outputRoles?.[`${name}.${x.name}`] ?? x.role } : {}),
      ...(o.readable?.[name] ? { readable: o.readable[name].includes(x.name) } : {}),
    })),
  };
}

// Constants blobs: hashed where they lie. They are the published artifact and
// they never enter git.
const constants = {};
for (const [name, src] of Object.entries(o.constants)) {
  if (!fs.existsSync(src)) throw new Error(`constants blob for "${name}" not found: ${src}`);
  const bytes = fs.statSync(src).size;
  process.stdout.write(`[add-entry] hashing ${path.basename(src)} (${(bytes / 2 ** 20).toFixed(0)} MiB)... `);
  const sha256 = sha256File(src);
  console.log(sha256.slice(0, 16) + "...");
  const recipe = readJson(path.join(entryDir, graphs[name].recipe));
  const usedBytes = Math.max(...Object.values(recipe.constants).map((c) => c.byteOffset + c.byteLength));
  constants[name] = {
    file: path.basename(src),
    bytes,
    ...(usedBytes !== bytes ? { usedBytes } : {}),
    sha256,
    url: null,
    chunks: null,
    ...(usedBytes !== bytes ? { note: `The trailing ${bytes - usedBytes} bytes are alignment padding; the recipe indexes ${usedBytes}.` } : {}),
  };
}

writeJson(path.join(entryDir, "manifest.json"), { entry: entryId, family: o.family, constants, recipeHashes });

// Verification set and provenance ledger, copied in verbatim.
if (o.verification) copyDirIfDifferent(o.verification, path.join(entryDir, "verification"));
if (o.provenance) copyIfDifferent(o.provenance, path.join(entryDir, "provenance.md"));

// Measurements.
if (o.measurements) {
  const row = rowFromResults(readJson(o.measurements), target, {
    label: o.measurementLabel,
    protocol: o.protocol,
    notes: o.measurementNotes,
    date: o.measurementDate,
    loadAvg: o.measurementLoadAvg,
    source: o.measurementSource ?? path.relative(path.dirname(ROOT), o.measurements),
    sameAsTarget: true,
  });
  // `--measurements-extra` carries the context that is not a row: the baselines
  // this entry was compared against, the floor and why, and the earlier passes.
  const extra = o.measurementsExtra ? readJson(o.measurementsExtra) : {};
  for (const k of Object.keys(extra))
    if (["entry", "family", "rows"].includes(k)) throw new Error(`--measurements-extra must not set "${k}"`);
  writeJson(path.join(entryDir, "measurements.json"), { entry: entryId, family: o.family, rows: [row], ...extra });
}

// ---------------------------------------------------------------------------
const requires = Object.keys(o.requires).length
  ? Object.fromEntries(Object.entries(o.requires).map(([k, val]) => [k, /^-?\d+$/.test(val) ? Number(val) : val]))
  : { "backend.name": target.backend.name };

const entry = {
  id: entryId,
  family: o.family,
  variant: o.variant,
  created: new Date().toISOString().slice(0, 10),
  ...(o.producedBy ? { producedBy: { workbench: o.producedBy, date: new Date().toISOString().slice(0, 10), ...(o.recorder ? { recorder: o.recorder } : {}) } } : {}),
  ...(o.title ? { title: o.title } : {}),
  ...(o.summary ? { summary: o.summary } : {}),
  target,
  compat: {
    requires,
    ops: [...allOps].sort(),
    notes:
      "Facts, not policy. `requires` is what must hold for these graphs to build and compute correctly; " +
      "`target` additionally records where they were tuned and timed, which is a weaker claim. Timings apply " +
      "only to the hosts measurements.json names.",
  },
  graphs: { ...graphs, chain: o.chain },
  constants: "manifest.json",
  ...(o.verification ? { verification: "verification/" } : {}),
  ...(o.measurements ? { measurements: "measurements.json" } : {}),
  ...(o.provenance ? { provenance: "provenance.md" } : {}),
  ...(family.tokenizer ? { tokenizerFromFamily: true } : {}),
};
writeJson(path.join(entryDir, "entry.json"), entry);

// ---------------------------------------------------------------------------
// Catalog index: a summary row, and nothing that is not a copy of the entry.
const catalogPath = path.join(ROOT, "catalog.json");
const catalog = fs.existsSync(catalogPath) && readJson(catalogPath).version === 2
  ? readJson(catalogPath)
  : {
      catalog: "webnn-catalog",
      version: 2,
      description: "Hand-built WebNN graphs, stored as data and keyed by the configuration they were built for. The catalog stores and describes; it does not select.",
      schema: { dir: "schema/", catalog: "schema/catalog.schema.json" },
      runtime: { loader: "runtime/loader.js", docs: "runtime/README.md", recipeSchemaVersion: 1 },
      families: {},
      conventions: {},
    };
catalog.generatedAt = new Date().toISOString();

// The conventions are part of the index: a consumer that reads only this file
// should still be told what the fields mean and what they do not promise.
const CONVENTIONS = {
  selection: "The catalog stores and describes; it does not select. There is no ranking, no fallback and no default entry here. Probing the machine and choosing an entry is the consuming product's policy.",
  entry: "One family built and tuned for one configuration. `target` is what it was built FOR, `compat.requires` is what must hold for it to work at all, and the two are different claims.",
  recipe: "The exact MLGraphBuilder call sequence, with the shapes the backend inferred. Not a model format. schema/recipe.schema.json is the contract; runtime/loader.js is one reader of it.",
  constants: "Never in git. A separate blob whose byteOffsets the recipe indexes, published to a content host and pinned by sha256 in the entry's manifest.",
  measurements: "One row per host, each naming the machine, the OS, the browser build and the protocol. A number without those is not a measurement, and a row's load average is never borrowed from another moment.",
  provenance: "Every entry has a ledger: what was folded, what was rejected and by which number, and what the backend turned out to be like.",
  index: "Summary rows only. The entry directory is authoritative for every field copied here.",
};
for (const [k, val] of Object.entries(CONVENTIONS)) catalog.conventions[k] ??= val;

const measurements = o.measurements ? readJson(path.join(entryDir, "measurements.json")) : { rows: [] };
const row = {
  id: entryId,
  path: path.relative(ROOT, entryDir),
  entry: path.relative(ROOT, path.join(entryDir, "entry.json")),
  variant: o.variant,
  backend: target.backend.name,
  browser: `${target.browser.name} ${target.browser.major}`,
  os: `${target.os.name} ${target.os.major}`,
  gpu: [target.gpu.vendor, target.gpu.architecture].filter(Boolean).join("/"),
  chip: target.host?.chip ?? null,
  created: entry.created,
  graphs: Object.keys(graphs),
  newPromptMs: measurements.rows[0]?.newPromptMs ?? null,
  measuredHosts: new Set(measurements.rows.map((r) => r.host.chip ?? r.host.browser)).size,
  constantBytes: Object.values(constants).reduce((a, c) => a + c.bytes, 0),
  weightsPublished: Object.values(constants).every((c) => c.url !== null),
};

catalog.families[o.family] ??= {
  path: path.relative(ROOT, famDir),
  family: path.relative(ROOT, famFile),
  name: family.name,
  task: family.task,
  ...(family.source?.model ? { model: family.source.model } : {}),
  entries: [],
};
const fam = catalog.families[o.family];
fam.name = family.name;
fam.task = family.task;
// Rows whose entry directory is gone are stale: the index is a copy of what is
// on disk, so it never keeps something disk does not have.
const pruned = fam.entries.filter((e) => !fs.existsSync(path.join(ROOT, e.entry)));
fam.entries = fam.entries.filter((e) => fs.existsSync(path.join(ROOT, e.entry)));
for (const p of pruned) console.log(`[add-entry] dropped stale index row ${o.family}/${p.id} (no ${p.entry})`);
const at = fam.entries.findIndex((e) => e.id === entryId);
if (at >= 0) fam.entries[at] = row; else fam.entries.push(row);
fam.entries.sort((a, b) => a.id.localeCompare(b.id));
writeJson(catalogPath, catalog);

// ---------------------------------------------------------------------------
console.log(`\nwrote ${path.relative(ROOT, entryDir)}/`);
for (const f of fs.readdirSync(entryDir)) console.log(`  ${f}`);
console.log(`updated catalog.json`);

const { errors } = validateCatalog({ only: `${o.family}/${entryId}`, quiet: true });
if (errors.length) {
  console.error(`\nthe entry was written but does NOT validate:`);
  for (const e of errors) console.error(`  - ${e}`);
  process.exitCode = 1;
} else {
  console.log(`validates against schema/`);
}
