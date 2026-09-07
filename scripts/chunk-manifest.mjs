#!/usr/bin/env node
// Write `chunks` into an entry's manifest.json, so a reader of a Range-served
// blob holds one chunk at a time instead of the whole thing.
//
//   node scripts/chunk-manifest.mjs [--entry <family>/<entry-id>] [--target 64MiB]
//                                   [--dry-run] [--clear]
//
// Without --entry every entry in the catalog is chunked.
//
// WHY THIS EXISTS
//
// `runtime/loader.js` reads `manifest.constants[key].chunks` and, when it is
// there, issues one ranged request per chunk and releases each before fetching
// the next. With `chunks: null` it fetches the whole file, which for the
// SD-Turbo image blob is a single 1.65 GiB ArrayBuffer held while Chrome is
// also copying the same bytes into the graph. The chunked path is the one a
// published, Range-served blob wants; this script is what fills it in.
//
// WHAT A BOUNDARY MAY BE
//
// The loader tolerates a constant straddling a chunk boundary -- it carries the
// head and stitches the tail out of the next chunk -- but a straddle costs a
// copy and loses the zero-copy view into the fetched bytes. So boundaries here
// are always constant START offsets, and only offsets that start a constant in
// EVERY recipe that reads the blob (the IntelliTeX decode blob is read by three
// recipes at three sequence lengths). A constant larger than the target simply
// gets a chunk of its own.
//
// The chunks tile [0, bytes) with no gaps, which is what buildConstants' walk
// requires: it advances through the constants in byteOffset order as the ranges
// arrive, and a gap would strand every constant behind it.

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const parseSize = (s) => {
  const m = /^(\d+(?:\.\d+)?)\s*(B|KB|KIB|MB|MIB|GB|GIB)?$/i.exec(String(s).trim());
  if (!m) throw new Error(`unparseable size "${s}"`);
  const unit = (m[2] ?? "B").toUpperCase();
  const mul = { B: 1, KB: 1e3, KIB: 1024, MB: 1e6, MIB: 1024 ** 2, GB: 1e9, GIB: 1024 ** 3 }[unit];
  return Math.round(Number(m[1]) * mul);
};

const args = process.argv.slice(2);
let only = null;
let target = 64 * 1024 * 1024;
let dryRun = false;
let clear = false;
for (let i = 0; i < args.length; i++) {
  switch (args[i]) {
    case "--entry": only = args[++i]; break;
    case "--target": target = parseSize(args[++i]); break;
    case "--dry-run": dryRun = true; break;
    case "--clear": clear = true; break;
    case "-h": case "--help":
      console.log(fs.readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n").slice(1, 8).join("\n").replace(/^\/\/ ?/gm, ""));
      process.exit(0);
    default: throw new Error(`unknown argument: ${args[i]}`);
  }
}

const readJson = (p) => JSON.parse(fs.readFileSync(p, "utf8"));
const mib = (n) => `${(n / 1024 ** 2).toFixed(1)} MiB`;

/** Entries to touch, as {family, id, dir}. */
function entries() {
  const out = [];
  for (const family of fs.readdirSync(path.join(ROOT, "families")).sort()) {
    const entriesDir = path.join(ROOT, "families", family, "entries");
    if (!fs.existsSync(entriesDir)) continue;
    for (const id of fs.readdirSync(entriesDir).sort()) {
      if (only && only !== `${family}/${id}`) continue;
      out.push({ family, id, dir: path.join(entriesDir, id) });
    }
  }
  if (only && !out.length) throw new Error(`no entry ${only}`);
  return out;
}

/**
 * Offsets that start a constant in every recipe reading this blob, sorted.
 * Always includes 0, which is where the tiling starts whatever lives there.
 */
function boundaries(recipes) {
  let set = null;
  for (const r of recipes) {
    const here = new Set(Object.values(r.constants).map((c) => c.byteOffset));
    set = set === null ? here : new Set([...set].filter((o) => here.has(o)));
  }
  set ??= new Set();
  set.add(0);
  return [...set].sort((a, b) => a - b);
}

/**
 * Greedy tiling of [0, bytes): cut at the furthest boundary that keeps the
 * chunk within `target`. When even the next boundary is already past the
 * target, cut there anyway -- a constant bigger than the target is a chunk of
 * its own, and nothing can be done about that from here. Past the last
 * boundary the rest of the blob is one chunk, however big its tail constant is.
 */
function tile(bounds, bytes, targetBytes) {
  const chunks = [];
  let start = 0;
  while (start < bytes) {
    if (bytes - start <= targetBytes) { chunks.push({ byteOffset: start, byteLength: bytes - start }); break; }
    const limit = start + targetBytes;
    let cut = 0;
    for (const b of bounds) {
      if (b <= start) continue;
      if (b <= limit) cut = b; // furthest boundary that still fits
      else { if (!cut) cut = b; break; } // one oversized constant
    }
    if (!cut || cut > bytes) cut = bytes;
    chunks.push({ byteOffset: start, byteLength: cut - start });
    start = cut;
  }

  // A boundary can fall right after a tiny constant (Texify's decode blob opens
  // with an 8-byte scalar, then a 102 MB embedding matrix), and a request for 8
  // bytes is all latency and no payload. Drop any cut that leaves a chunk under
  // the floor, merging it into its neighbour; the survivor still starts on a
  // constant.
  const floor = Math.min(1024 ** 2, targetBytes);
  for (let i = 0; i < chunks.length; i++) {
    if (chunks[i].byteLength >= floor || chunks.length === 1) continue;
    if (i + 1 < chunks.length) {
      chunks[i].byteLength += chunks[i + 1].byteLength;
      chunks.splice(i + 1, 1);
    } else {
      chunks[i - 1].byteLength += chunks[i].byteLength;
      chunks.splice(i, 1);
    }
    i--;
  }
  return chunks;
}

/** Everything buildConstants assumes, checked here so a bad tiling never ships. */
function check(key, chunks, bytes, recipes) {
  let at = 0;
  for (const c of chunks) {
    if (c.byteOffset !== at) throw new Error(`${key}: chunk at ${c.byteOffset} leaves a gap after ${at}`);
    if (c.byteLength <= 0) throw new Error(`${key}: empty chunk at ${c.byteOffset}`);
    at += c.byteLength;
  }
  if (at !== bytes) throw new Error(`${key}: chunks cover ${at} of ${bytes} bytes`);
  const starts = new Set(chunks.map((c) => c.byteOffset));
  for (const r of recipes) {
    for (const [n, c] of Object.entries(r.constants)) {
      const chunk = chunks.findLast((k) => k.byteOffset <= c.byteOffset);
      if (c.byteOffset + c.byteLength > chunk.byteOffset + chunk.byteLength)
        throw new Error(`${key}: constant ${n} straddles the boundary at ${chunk.byteOffset + chunk.byteLength}`);
      if (c.byteOffset % 8 !== 0 && starts.has(c.byteOffset))
        throw new Error(`${key}: boundary ${c.byteOffset} is not 8-byte aligned`);
    }
  }
}

let touched = 0;
for (const e of entries()) {
  const entry = readJson(path.join(e.dir, "entry.json"));
  // Only WebNN recipe entries index a constants blob by byte offset. A WebLLM
  // entry names hash-pinned artifacts instead, and has nothing to chunk.
  if ((entry.runtimeKind ?? "webnn") !== "webnn") continue;
  const manifestPath = path.join(e.dir, entry.constants ?? "manifest.json");
  const manifest = readJson(manifestPath);

  // Which recipes read each blob. A blob shared by several graphs (IntelliTeX's
  // decode blob is read at three sequence lengths) must satisfy all of them.
  const readers = new Map();
  for (const [g, spec] of Object.entries(entry.graphs)) {
    if (g === "chain") continue;
    const list = readers.get(spec.constants) ?? [];
    list.push({ graph: g, recipe: readJson(path.join(e.dir, spec.recipe)) });
    readers.set(spec.constants, list);
  }

  console.log(`${e.family}/${e.id}`);
  for (const [key, record] of Object.entries(manifest.constants)) {
    const list = readers.get(key);
    if (!list) { console.log(`  ${key}: no graph reads it, left alone`); continue; }
    const recipes = list.map((l) => l.recipe);

    if (clear) {
      record.chunks = null;
      console.log(`  ${key}: chunks cleared`);
      touched++;
      continue;
    }

    const used = Math.max(...recipes.flatMap((r) => Object.values(r.constants).map((c) => c.byteOffset + c.byteLength)));
    const chunks = tile(boundaries(recipes), record.bytes, target);
    check(key, chunks, record.bytes, recipes);

    const largest = Math.max(...chunks.map((c) => c.byteLength));
    record.chunks = chunks.length > 1 ? chunks : null;
    if (used < record.bytes) record.usedBytes = used;
    else delete record.usedBytes;

    const read = list.map((l) => l.graph).join(", ");
    const over = largest > target ? `, over target because one constant is that big` : "";
    console.log(
      chunks.length > 1
        ? `  ${key}: ${mib(record.bytes)} -> ${chunks.length} chunks, largest ${mib(largest)}${over} (read by ${read})`
        : `  ${key}: ${mib(record.bytes)} fits one chunk, left whole (read by ${read})`,
    );
    touched++;
  }

  // The manifest pins each recipe's sha256; chunking must not have touched one.
  for (const [name, rec] of Object.entries(manifest.recipeHashes ?? {})) {
    const file = path.join(e.dir, name);
    if (!fs.existsSync(file)) throw new Error(`recipeHashes names ${name}, which is not here`);
    const sha = createHash("sha256").update(fs.readFileSync(file)).digest("hex");
    if (sha !== rec.sha256) throw new Error(`${name}: sha256 ${sha} != manifest ${rec.sha256}`);
  }

  if (dryRun) console.log("  [dry-run] manifest unchanged");
  else fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
}

console.log(`\n${touched} blob${touched === 1 ? "" : "s"} ${dryRun ? "planned" : clear ? "cleared" : "chunked"}`);
