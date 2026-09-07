#!/usr/bin/env node
// Replay every graph in the catalog through runtime/loader.js with a stub
// MLGraphBuilder, driving the constants off the manifest's `chunks`, and check
// that the chunked path builds every constant the recipe names.
//
//   node scripts/check-chunks.mjs [--entry <family>/<entry-id>] [--http]
//
// Offline (the default) the chunks are served out of a zero-filled blob: this
// exercises the walk in buildConstants -- coverage, ordering, straddles, and
// the alignment of every view handed to builder.constant() -- without moving a
// byte over the network. What it cannot check is that the ranges are the ranges
// of the published file.
//
// --http fetches the chunks from each blob's published `url` for real and
// sha256s them back as they arrive. Because the chunks tile the blob exactly,
// their concatenation must hash to the manifest's sha256: that is the whole
// claim, that this chunk list reconstructs the published bytes and nothing
// else. It downloads the entry, so give it an --entry.
//
// This is deliberately not part of validate.mjs, which is a static check over
// files in git.

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { constantSource, loadRecipe } from "../runtime/loader.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const args = process.argv.slice(2);
let only = null;
let http = false;
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--entry") only = args[++i];
  else if (args[i] === "--http") http = true;
  else throw new Error(`unknown argument: ${args[i]}`);
}

const readJson = (p) => JSON.parse(fs.readFileSync(p, "utf8"));
const mib = (n) => `${(n / 1024 ** 2).toFixed(1)} MiB`;

/**
 * Enough of MLGraphBuilder to replay a recipe on a machine with no WebNN.
 *
 * It records what it was handed rather than computing anything: the point is
 * the plumbing, not the arithmetic. `constant` is where the checking happens --
 * every view must have the byte length the descriptor implies, and be aligned
 * enough that the loader did not have to fall back to a copy.
 */
function stubBuilder(recipe, seen) {
  const width = { float32: 4, float16: 2, int32: 4, uint32: 4, int64: 8, uint64: 8, int8: 1, uint8: 1 };
  const token = (kind) => ({ __stub: kind });
  const handler = {
    get(_t, prop) {
      if (prop === "constant") {
        return (desc, view) => {
          const n = desc.shape.reduce((a, b) => a * b, 1);
          const want = n * width[desc.dataType];
          if (view.byteLength !== want)
            throw new Error(`constant view is ${view.byteLength} bytes, descriptor ${desc.dataType}${JSON.stringify(desc.shape)} wants ${want}`);
          const elem = view.BYTES_PER_ELEMENT ?? 1;
          if (view.byteOffset % elem !== 0) throw new Error(`constant view misaligned at byteOffset ${view.byteOffset}`);
          seen.constants++;
          seen.constantBytes += view.byteLength;
          return token("constant");
        };
      }
      if (prop === "input") return () => token("input");
      if (prop === "build") return async () => ({ __stub: "graph" });
      if (prop === "split") {
        return (_input, splits) => {
          const n = Array.isArray(splits) ? splits.length : splits;
          return Array.from({ length: n }, () => token("split"));
        };
      }
      return (...a) => { void a; seen.ops++; return token(String(prop)); };
    },
  };
  void recipe;
  return new Proxy({}, handler);
}

/** A source with the manifest's ranges, answered out of zeros. */
function zeroSource(record) {
  return {
    kind: "zeros",
    totalBytes: record.bytes,
    ranges: record.chunks ?? [{ byteOffset: 0, byteLength: record.bytes }],
    async fetchRange(byteOffset, byteLength) {
      void byteOffset;
      return new Uint8Array(byteLength);
    },
  };
}

/** The real published source, hashing every byte it hands back. */
function hashingSource(record) {
  const inner = constantSource(record);
  const hash = createHash("sha256");
  let read = 0;
  return {
    kind: inner.kind,
    totalBytes: inner.totalBytes,
    ranges: inner.ranges,
    digest: () => hash.digest("hex"),
    bytesRead: () => read,
    async fetchRange(byteOffset, byteLength) {
      const bytes = await inner.fetchRange(byteOffset, byteLength);
      hash.update(bytes);
      read += bytes.byteLength;
      return bytes;
    },
  };
}

const entries = [];
for (const family of fs.readdirSync(path.join(ROOT, "families")).sort()) {
  const dir = path.join(ROOT, "families", family, "entries");
  if (!fs.existsSync(dir)) continue;
  for (const id of fs.readdirSync(dir).sort()) {
    if (only && only !== `${family}/${id}`) continue;
    entries.push({ family, id, dir: path.join(dir, id) });
  }
}
if (only && !entries.length) throw new Error(`no entry ${only}`);
if (http && !only) throw new Error("--http downloads the weights: name one entry with --entry <family>/<entry-id>");

let failures = 0;
let graphs = 0;
for (const e of entries) {
  const entry = readJson(path.join(e.dir, "entry.json"));
  // WebLLM entries have no constants blob to tile; skip them.
  if ((entry.runtimeKind ?? "webnn") !== "webnn") continue;
  const manifest = readJson(path.join(e.dir, entry.constants ?? "manifest.json"));
  console.log(`${e.family}/${e.id}`);

  const digests = new Map(); // blob key -> the source that hashed it

  for (const [name, spec] of Object.entries(entry.graphs)) {
    if (name === "chain") continue;
    const recipe = readJson(path.join(e.dir, spec.recipe));
    const record = manifest.constants[spec.constants];
    const source = http ? hashingSource(record) : zeroSource(record);
    const seen = { constants: 0, constantBytes: 0, ops: 0 };
    const want = Object.keys(recipe.constants).length;

    try {
      const loaded = await loadRecipe(recipe, source, null, { builder: stubBuilder(recipe, seen) });
      if (seen.constants !== want) throw new Error(`built ${seen.constants} of ${want} constants`);
      if (Object.keys(loaded.inputs).length !== recipe.inputs.length) throw new Error("input count changed");
      const how = record.chunks ? `${record.chunks.length} chunks, peak ${mib(Math.max(...record.chunks.map((c) => c.byteLength)))}` : "whole blob";
      console.log(`  ok  ${name}: ${want} constants, ${recipe.ops.length} ops, ${how}`);
      if (http) digests.set(spec.constants, source);
      graphs++;
    } catch (err) {
      console.log(`  FAIL ${name}: ${err.message}`);
      failures++;
    }
  }

  if (http) {
    for (const [key, source] of digests) {
      const record = manifest.constants[key];
      const got = source.digest();
      const ok = got === record.sha256 && source.bytesRead() === record.bytes;
      console.log(`  ${ok ? "ok " : "FAIL"} ${key}: fetched ${source.bytesRead()}/${record.bytes} bytes, sha256 ${got.slice(0, 16)}... ${ok ? "matches" : `!= ${record.sha256.slice(0, 16)}...`}`);
      if (!ok) failures++;
    }
  }
}

console.log(`\n${graphs} graph${graphs === 1 ? "" : "s"} replayed${http ? " over the published chunks" : " offline"}`);
if (failures) { console.log(`${failures} FAILED`); process.exit(1); }
console.log("CHUNKS OK");
