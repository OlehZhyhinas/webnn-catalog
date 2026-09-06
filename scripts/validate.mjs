// Validate the whole catalog against schema/, plus the cross-file facts a
// schema cannot express.
//
//   node scripts/validate.mjs [--quiet] [--entry <family>/<entry-id>]
//
// Two layers:
//
//   1. JSON Schema (draft 2020-12, the subset the schemas use) for every file:
//      catalog.json, family.json, entry.json, manifest.json, measurements.json,
//      every recipe, and each entry's `target` block.
//   2. Cross-file checks: referenced paths exist, recipe files hash to what the
//      manifest says, a graph's `constants` key names a real blob, the entry's
//      declared inputs/outputs equal the recipe's, operands are defined before
//      use, constant byteLengths agree with their shapes, and the catalog index
//      agrees with the entries it indexes.
//
// The validator is deliberately dependency-free: `npm install` here pulls
// Playwright for scripts/verify.mjs and nothing else. It implements the subset
// of JSON Schema the schemas in schema/ actually use, and REFUSES a schema that
// uses a keyword it does not know rather than silently passing it.

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(HERE, "..");
const SCHEMA_DIR = path.join(ROOT, "schema");

// ---------------------------------------------------------------------------
// A small JSON Schema validator
// ---------------------------------------------------------------------------

const KNOWN = new Set([
  "$schema", "$id", "$ref", "$defs", "$comment", "title", "description", "default", "examples",
  "type", "enum", "const", "required", "properties", "patternProperties", "additionalProperties",
  "propertyNames", "minProperties", "maxProperties", "items", "prefixItems", "minItems", "maxItems",
  "uniqueItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
  "minLength", "maxLength", "pattern", "format", "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
]);

const typeOf = (v) =>
  v === null ? "null" : Array.isArray(v) ? "array" : Number.isInteger(v) ? "integer" : typeof v;

const matchesType = (v, t) =>
  t === "integer" ? Number.isInteger(v)
  : t === "number" ? typeof v === "number"
  : t === "array" ? Array.isArray(v)
  : t === "object" ? v !== null && typeof v === "object" && !Array.isArray(v)
  : t === "null" ? v === null
  : typeof v === t;

class Registry {
  constructor(dir) {
    this.dir = dir;
    this.byFile = new Map();
    for (const f of fs.readdirSync(dir).filter((f) => f.endsWith(".schema.json")))
      this.byFile.set(f, JSON.parse(fs.readFileSync(path.join(dir, f), "utf8")));
  }
  get(name) {
    const s = this.byFile.get(name);
    if (!s) throw new Error(`no schema ${name} in ${this.dir}`);
    return s;
  }
  /** Resolve "#/$defs/x", "other.schema.json", "other.schema.json#/$defs/x". */
  resolve(ref, currentRoot, currentFile) {
    const [file, pointer] = ref.split("#");
    const root = file ? this.get(file) : currentRoot;
    const rootFile = file || currentFile;
    let node = root;
    if (pointer) {
      for (const raw of pointer.split("/").slice(1)) {
        const seg = decodeURIComponent(raw).replace(/~1/g, "/").replace(/~0/g, "~");
        node = node?.[seg];
        if (node === undefined) throw new Error(`unresolvable $ref ${ref}`);
      }
    }
    return { schema: node, root, file: rootFile };
  }
}

function check(schema, data, ctx, at, errors) {
  if (schema === true || schema === undefined) return;
  if (schema === false) return void errors.push(`${at}: schema forbids any value here`);

  for (const k of Object.keys(schema))
    if (!KNOWN.has(k) && !k.startsWith("x-")) errors.push(`${at}: schema uses unsupported keyword "${k}"`);

  if (schema.$ref) {
    const r = ctx.reg.resolve(schema.$ref, ctx.root, ctx.file);
    check(r.schema, data, { ...ctx, root: r.root, file: r.file }, at, errors);
    // Sibling keywords next to $ref still apply in 2020-12.
  }

  const fail = (msg) => errors.push(`${at}: ${msg}`);

  if (schema.type !== undefined) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((t) => matchesType(data, t)))
      return void fail(`expected ${types.join("|")}, got ${typeOf(data)}`);
  }
  if (schema.const !== undefined && JSON.stringify(data) !== JSON.stringify(schema.const))
    fail(`must be ${JSON.stringify(schema.const)}, got ${JSON.stringify(data)}`);
  if (schema.enum !== undefined && !schema.enum.some((e) => JSON.stringify(e) === JSON.stringify(data)))
    fail(`must be one of ${schema.enum.map((e) => JSON.stringify(e)).join(", ")}, got ${JSON.stringify(data)}`);

  if (typeof data === "string") {
    if (schema.minLength !== undefined && data.length < schema.minLength) fail(`shorter than ${schema.minLength}`);
    if (schema.maxLength !== undefined && data.length > schema.maxLength) fail(`longer than ${schema.maxLength}`);
    if (schema.pattern !== undefined && !new RegExp(schema.pattern).test(data))
      fail(`"${data}" does not match /${schema.pattern}/`);
  }

  if (typeof data === "number") {
    if (schema.minimum !== undefined && data < schema.minimum) fail(`${data} < minimum ${schema.minimum}`);
    if (schema.maximum !== undefined && data > schema.maximum) fail(`${data} > maximum ${schema.maximum}`);
    if (schema.exclusiveMinimum !== undefined && data <= schema.exclusiveMinimum) fail(`${data} <= exclusiveMinimum`);
    if (schema.exclusiveMaximum !== undefined && data >= schema.exclusiveMaximum) fail(`${data} >= exclusiveMaximum`);
    if (schema.multipleOf !== undefined && data % schema.multipleOf !== 0) fail(`${data} is not a multiple of ${schema.multipleOf}`);
  }

  if (Array.isArray(data)) {
    if (schema.minItems !== undefined && data.length < schema.minItems) fail(`needs at least ${schema.minItems} items`);
    if (schema.maxItems !== undefined && data.length > schema.maxItems) fail(`allows at most ${schema.maxItems} items`);
    if (schema.uniqueItems && new Set(data.map((d) => JSON.stringify(d))).size !== data.length) fail(`items must be unique`);
    const prefix = schema.prefixItems ?? [];
    data.forEach((v, i) => {
      if (i < prefix.length) check(prefix[i], v, ctx, `${at}[${i}]`, errors);
      else if (schema.items !== undefined) check(schema.items, v, ctx, `${at}[${i}]`, errors);
    });
  }

  if (data !== null && typeof data === "object" && !Array.isArray(data)) {
    const keys = Object.keys(data);
    for (const r of schema.required ?? []) if (!(r in data)) fail(`missing required property "${r}"`);
    if (schema.minProperties !== undefined && keys.length < schema.minProperties) fail(`needs at least ${schema.minProperties} properties`);
    if (schema.maxProperties !== undefined && keys.length > schema.maxProperties) fail(`allows at most ${schema.maxProperties} properties`);
    for (const k of keys) {
      let covered = false;
      if (schema.properties && k in schema.properties) {
        covered = true;
        check(schema.properties[k], data[k], ctx, `${at}.${k}`, errors);
      }
      for (const [pat, sub] of Object.entries(schema.patternProperties ?? {})) {
        if (new RegExp(pat).test(k)) {
          covered = true;
          check(sub, data[k], ctx, `${at}.${k}`, errors);
        }
      }
      if (!covered && schema.additionalProperties !== undefined) {
        if (schema.additionalProperties === false) fail(`unexpected property "${k}"`);
        else check(schema.additionalProperties, data[k], ctx, `${at}.${k}`, errors);
      }
      if (schema.propertyNames !== undefined) check(schema.propertyNames, k, ctx, `${at} key "${k}"`, errors);
    }
  }

  for (const sub of schema.allOf ?? []) check(sub, data, ctx, at, errors);
  if (schema.anyOf) {
    const ok = schema.anyOf.some((sub) => { const e = []; check(sub, data, ctx, at, e); return e.length === 0; });
    if (!ok) fail(`matches none of the anyOf alternatives`);
  }
  if (schema.oneOf) {
    const n = schema.oneOf.filter((sub) => { const e = []; check(sub, data, ctx, at, e); return e.length === 0; }).length;
    if (n !== 1) fail(`matches ${n} of the oneOf alternatives, expected exactly 1`);
  }
  if (schema.not) {
    const e = [];
    check(schema.not, data, ctx, at, e);
    if (e.length === 0) fail(`must NOT match the "not" schema`);
  }
  if (schema.if) {
    const e = [];
    check(schema.if, data, ctx, at, e);
    const branch = e.length === 0 ? schema.then : schema.else;
    if (branch) check(branch, data, ctx, at, errors);
  }
}

/** A validator bound to schema/. `validate(schemaFile, data, label) -> string[]` */
export function createValidator(dir = SCHEMA_DIR) {
  const reg = new Registry(dir);
  return {
    schemas: reg,
    validate(schemaFile, data, label = schemaFile.replace(".schema.json", "")) {
      const root = reg.get(schemaFile);
      const errors = [];
      check(root, data, { reg, root, file: schemaFile }, label, errors);
      return errors;
    },
  };
}

// ---------------------------------------------------------------------------
// Cross-file checks
// ---------------------------------------------------------------------------

const readJson = (p) => JSON.parse(fs.readFileSync(p, "utf8"));
const sha256File = (p) => crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
const artifactBytes = (manifest) =>
  ["runtimeJs", "modelWasm", "modelResources"]
    .flatMap((key) => manifest[key] ?? [])
    .reduce((sum, artifact) => sum + (artifact.bytes ?? 0), 0);

function isPublicHttpsUrl(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" || url.username || url.password) return false;
    const host = url.hostname.toLowerCase().replace(/^\[|\]$/g, "");
    if (!host || host === "localhost" || host.endsWith(".localhost") || host.endsWith(".local")) return false;
    if (host === "::1" || host.startsWith("fc") || host.startsWith("fd") || host.startsWith("fe80:")) return false;
    const octets = host.split(".").map(Number);
    if (octets.length === 4 && octets.every((n) => Number.isInteger(n) && n >= 0 && n <= 255)) {
      if (octets[0] === 10 || octets[0] === 127 || octets[0] === 0) return false;
      if (octets[0] === 169 && octets[1] === 254) return false;
      if (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31) return false;
      if (octets[0] === 192 && octets[1] === 168) return false;
    }
    return true;
  } catch {
    return false;
  }
}

const BYTES_PER_ELEMENT = { float32: 4, float16: 2, int32: 4, uint32: 4, int64: 8, uint64: 8, int8: 1, uint8: 1 };

/** Structural checks a JSON Schema cannot make: the recipe must be replayable. */
export function checkRecipe(recipe, label) {
  const errors = [];
  const defined = new Set();
  for (const [name, c] of Object.entries(recipe.constants)) {
    const want = c.shape.reduce((a, b) => a * b, 1) * BYTES_PER_ELEMENT[c.dataType];
    if (want !== c.byteLength)
      errors.push(`${label}: constant ${name} is ${c.byteLength} bytes but ${c.dataType}${JSON.stringify(c.shape)} is ${want}`);
    defined.add(name);
  }
  for (const i of recipe.inputs) {
    if (defined.has(i.name)) errors.push(`${label}: input "${i.name}" collides with a constant`);
    defined.add(i.name);
  }
  let lastId = -1;
  for (const op of recipe.ops) {
    if (op.id !== lastId + 1) errors.push(`${label}: op ids are not dense: ${lastId} -> ${op.id}`);
    lastId = op.id;
    for (const n of op.inputs)
      if (!defined.has(n)) errors.push(`${label}: op ${op.id} (${op.type}) uses "${n}" before it is defined`);
    for (const [k, v] of Object.entries(op.options ?? {}))
      if (typeof v === "string" && defined.has(v) && !op.inputs.includes(v))
        errors.push(`${label}: op ${op.id} (${op.type}) option ${k} names operand "${v}" which is not in inputs`);
    const outs = op.outputs ?? [op.output];
    if (op.outputs && op.outputs[0] !== op.output)
      errors.push(`${label}: op ${op.id} outputs[0] "${op.outputs[0]}" is not output "${op.output}"`);
    if (op.outputShapes && JSON.stringify(op.outputShapes[0]) !== JSON.stringify(op.outputShape))
      errors.push(`${label}: op ${op.id} outputShapes[0] does not equal outputShape`);
    if (op.outputs && op.outputShapes && op.outputs.length !== op.outputShapes.length)
      errors.push(`${label}: op ${op.id} has ${op.outputs.length} outputs and ${op.outputShapes.length} outputShapes`);
    for (const n of outs) defined.add(n);
  }
  for (const o of recipe.outputs)
    if (!defined.has(o.operand)) errors.push(`${label}: output "${o.name}" names undefined operand "${o.operand}"`);
  // Offsets must be monotonic, or a chunked reader cannot release ranges.
  const offsets = Object.values(recipe.constants).map((c) => c.byteOffset);
  for (let i = 1; i < offsets.length; i++)
    if (offsets[i] < offsets[i - 1]) { errors.push(`${label}: constant byteOffsets are not monotonic`); break; }
  return errors;
}

const rel = (p) => path.relative(ROOT, p);

export function validateCatalog({ only = null, quiet = false } = {}) {
  const v = createValidator();
  const errors = [];
  const checked = [];
  const note = (s) => { checked.push(s); if (!quiet) console.log(`  ok  ${s}`); };
  const bad = (s) => errors.push(s);

  const catalogPath = path.join(ROOT, "catalog.json");
  const catalog = readJson(catalogPath);
  for (const e of v.validate("catalog.schema.json", catalog, "catalog.json")) bad(e);
  note("catalog.json");

  for (const [famId, famRow] of Object.entries(catalog.families)) {
    const famDir = path.join(ROOT, famRow.path);
    const famFile = path.join(ROOT, famRow.family);
    if (!fs.existsSync(famFile)) { bad(`catalog.json: families.${famId}.family -> missing ${famRow.family}`); continue; }
    const family = readJson(famFile);
    for (const e of v.validate("family.schema.json", family, rel(famFile))) bad(e);
    if (family.id !== famId) bad(`${rel(famFile)}: id "${family.id}" is not its catalog key "${famId}"`);
    if (family.name !== famRow.name) bad(`catalog.json: families.${famId}.name disagrees with family.json`);
    if (family.task !== famRow.task) bad(`catalog.json: families.${famId}.task disagrees with family.json`);
    if (family.runtimeKind !== famRow.runtimeKind) bad(`catalog.json: families.${famId}.runtimeKind disagrees with family.json`);
    for (const f of family.tokenizer?.files ?? [])
      if (!fs.existsSync(path.join(famDir, f))) bad(`${rel(famFile)}: tokenizer file ${f} is missing`);
    if (family.tokenizer?.module && !fs.existsSync(path.join(famDir, family.tokenizer.module)))
      bad(`${rel(famFile)}: tokenizer module ${family.tokenizer.module} is missing`);
    note(rel(famFile));

    const onDisk = fs.existsSync(path.join(famDir, "entries"))
      ? fs.readdirSync(path.join(famDir, "entries")).filter((d) => fs.statSync(path.join(famDir, "entries", d)).isDirectory())
      : [];
    const indexed = famRow.entries.map((e) => e.id);
    for (const d of onDisk) if (!indexed.includes(d)) bad(`catalog.json: entry ${famId}/${d} is on disk but not indexed`);

    for (const row of famRow.entries) {
      if (only && only !== `${famId}/${row.id}`) continue;
      const entryDir = path.join(ROOT, row.path);
      const entryFile = path.join(ROOT, row.entry);
      if (!fs.existsSync(entryFile)) { bad(`catalog.json: ${famId}/${row.id} -> missing ${row.entry}`); continue; }
      const entry = readJson(entryFile);
      for (const e of v.validate("entry.schema.json", entry, rel(entryFile))) bad(e);
      for (const e of v.validate("target.schema.json", entry.target, `${rel(entryFile)}.target`)) bad(e);

      if (entry.id !== row.id) bad(`${rel(entryFile)}: id "${entry.id}" is not its directory name "${row.id}"`);
      if (entry.id !== path.basename(entryDir)) bad(`${rel(entryFile)}: id does not match the directory it is in`);
      if (entry.family !== famId) bad(`${rel(entryFile)}: family "${entry.family}" is not "${famId}"`);
      if (entry.runtimeKind !== family.runtimeKind) bad(`${rel(entryFile)}: runtimeKind "${entry.runtimeKind}" disagrees with family.json`);
      if (entry.runtimeKind !== row.runtimeKind) bad(`catalog.json: ${famId}/${row.id}.runtimeKind disagrees with entry.json`);
      if (entry.variant !== row.variant) bad(`catalog.json: ${famId}/${row.id}.variant disagrees with entry.json`);
      if (entry.target.backend.name !== row.backend) bad(`catalog.json: ${famId}/${row.id}.backend disagrees with entry.json`);

      // compat.requires must name real paths in the target, with the values it has.
      for (const [dotted, want] of Object.entries(entry.compat.requires)) {
        const got = dotted.split(".").reduce((o, k) => (o === undefined || o === null ? o : o[k]), entry.target);
        if (got === undefined) bad(`${rel(entryFile)}: compat.requires["${dotted}"] names nothing in target`);
        else if (got !== want) bad(`${rel(entryFile)}: compat.requires["${dotted}"] is ${JSON.stringify(want)} but target says ${JSON.stringify(got)}`);
      }

      if (entry.runtimeKind === "webnn") {
        // WebNN manifest
        const manifestPath = path.join(entryDir, entry.constants);
        if (!fs.existsSync(manifestPath)) { bad(`${rel(entryFile)}: constants -> missing ${entry.constants}`); continue; }
        const manifest = readJson(manifestPath);
        for (const e of v.validate("manifest.schema.json", manifest, rel(manifestPath))) bad(e);
        if (manifest.entry !== entry.id || manifest.family !== entry.family)
          bad(`${rel(manifestPath)}: entry/family do not match entry.json`);
        const allPublished = Object.values(manifest.constants).every((c) => c.url !== null);
        if (row.weightsPublished !== allPublished)
          bad(`catalog.json: ${famId}/${row.id}.weightsPublished is ${row.weightsPublished} but manifest.json ${allPublished ? "has" : "lacks"} a url for every blob`);
        const totalBytes = Object.values(manifest.constants).reduce((sum, c) => sum + c.bytes, 0);
        if (row.constantBytes !== totalBytes)
          bad(`catalog.json: ${famId}/${row.id}.constantBytes disagrees with manifest.json`);
        if (row.artifactBytes !== totalBytes)
          bad(`catalog.json: ${famId}/${row.id}.artifactBytes disagrees with manifest.json`);

        // graphs: recipes exist, hash, and agree with the entry's declared I/O
        const graphNames = Object.keys(entry.graphs).filter((k) => k !== "chain");
        if (!graphNames.length) bad(`${rel(entryFile)}: no graphs`);
        const loaded = {};
        for (const g of graphNames) {
          const spec = entry.graphs[g];
          const recipePath = path.join(entryDir, spec.recipe);
          if (!fs.existsSync(recipePath)) { bad(`${rel(entryFile)}: graphs.${g}.recipe -> missing ${spec.recipe}`); continue; }
          const recipe = readJson(recipePath);
          loaded[g] = recipe;
          for (const e of v.validate("recipe.schema.json", recipe, rel(recipePath))) bad(e);
          for (const e of checkRecipe(recipe, rel(recipePath))) bad(e);
          const hash = manifest.recipeHashes?.[spec.recipe]?.sha256;
          if (!hash) bad(`${rel(manifestPath)}: no recipeHashes entry for ${spec.recipe}`);
          else if (hash !== sha256File(recipePath))
            bad(`${rel(recipePath)}: sha256 does not match manifest.recipeHashes (the file was edited by hand?)`);
          if (!manifest.constants[spec.constants])
            bad(`${rel(entryFile)}: graphs.${g}.constants "${spec.constants}" is not in manifest.json`);
          const same = (a, b) =>
            a.length === b.length &&
            a.every((x, i) => x.name === b[i].name && x.dataType === b[i].dataType && String(x.shape) === String(b[i].shape));
          if (!same(spec.inputs, recipe.inputs.map((i) => ({ name: i.name, dataType: i.dataType, shape: i.shape }))))
            bad(`${rel(entryFile)}: graphs.${g}.inputs disagree with ${spec.recipe}`);
          if (!same(spec.outputs.map((o) => ({ name: o.name, dataType: o.dataType, shape: o.shape })),
                    recipe.outputs.map((o) => ({ name: o.name, dataType: o.dataType, shape: o.shape }))))
            bad(`${rel(entryFile)}: graphs.${g}.outputs disagree with ${spec.recipe}`);
          if (spec.ops !== undefined && spec.ops !== recipe.ops.length)
            bad(`${rel(entryFile)}: graphs.${g}.ops says ${spec.ops}, the recipe has ${recipe.ops.length}`);
          note(rel(recipePath));
        }

        // chain links must name a real output and a real input of the same shape
        for (const [from, to] of entry.graphs.chain) {
          const [fg, fo] = from.split(".");
          const [tg, ti] = to.split(".");
          const out = entry.graphs[fg]?.outputs?.find((o) => o.name === fo);
          const inp = entry.graphs[tg]?.inputs?.find((i) => i.name === ti);
          if (!out) bad(`${rel(entryFile)}: chain "${from}" names no such graph output`);
          if (!inp) bad(`${rel(entryFile)}: chain "${to}" names no such graph input`);
          if (out && inp && (out.dataType !== inp.dataType || String(out.shape) !== String(inp.shape)))
            bad(`${rel(entryFile)}: chain ${from} -> ${to} does not typecheck`);
        }

        // family contract: the entry must actually implement it
        for (const [g, c] of Object.entries(family.contract.graphs)) {
          const spec = entry.graphs[g];
          if (!spec) { bad(`${rel(entryFile)}: family contract declares graph "${g}" and the entry has none`); continue; }
          for (const [name, want] of Object.entries(c.inputs)) {
            const got = spec.inputs.find((i) => i.name === name);
            if (!got) bad(`${rel(entryFile)}: graphs.${g} is missing contract input "${name}"`);
            else if (got.dataType !== want.dataType || String(got.shape) !== String(want.shape))
              bad(`${rel(entryFile)}: graphs.${g}.${name} is ${got.dataType}[${got.shape}], the family contract says ${want.dataType}[${want.shape}]`);
          }
          for (const [name, want] of Object.entries(c.outputs)) {
            const got = spec.outputs.find((o) => o.name === name);
            if (!got) bad(`${rel(entryFile)}: graphs.${g} is missing contract output "${name}"`);
            else if (got.dataType !== want.dataType || String(got.shape) !== String(want.shape))
              bad(`${rel(entryFile)}: graphs.${g}.${name} is ${got.dataType}[${got.shape}], the family contract says ${want.dataType}[${want.shape}]`);
          }
        }
      } else if (entry.runtimeKind === "webllm") {
        const manifestPath = path.join(entryDir, entry.artifacts);
        if (!fs.existsSync(manifestPath)) { bad(`${rel(entryFile)}: artifacts -> missing ${entry.artifacts}`); continue; }
        const manifest = readJson(manifestPath);
        for (const e of v.validate("artifact-manifest.schema.json", manifest, rel(manifestPath))) bad(e);
        if (manifest.entry !== entry.id || manifest.family !== entry.family)
          bad(`${rel(manifestPath)}: entry/family do not match entry.json`);
        for (const kind of ["runtimeJs", "modelWasm", "modelResources"]) {
          for (const artifact of manifest[kind] ?? []) {
            if (!isPublicHttpsUrl(artifact.url))
              bad(`${rel(manifestPath)}: ${kind}.${artifact.name ?? "?"}.url must be a public HTTPS URL`);
            if (kind !== "modelResources" && !/^[0-9a-f]{64}$/.test(artifact.sha256 ?? ""))
              bad(`${rel(manifestPath)}: ${kind}.${artifact.name ?? "?"}.sha256 must be 64 lowercase hex characters`);
            if (!Number.isInteger(artifact.bytes) || artifact.bytes <= 0)
              bad(`${rel(manifestPath)}: ${kind}.${artifact.name ?? "?"}.bytes must be a positive integer`);
          }
        }
        for (const resource of manifest.modelResources ?? []) {
          if (resource.revision !== entry.model?.revision)
            bad(`${rel(manifestPath)}: model resource revision disagrees with entry.model.revision`);
        }
        if (!isPublicHttpsUrl(entry.source?.repository) || !entry.source?.commit)
          bad(`${rel(entryFile)}: WebLLM source requires a public HTTPS repository and pinned commit`);
        if (!entry.license?.name)
          bad(`${rel(entryFile)}: WebLLM license metadata is required`);
        if (!entry.model?.revision)
          bad(`${rel(entryFile)}: WebLLM model revision is required`);
        if (typeof entry.capabilities?.fastPath?.eligible !== "boolean" || !entry.capabilities?.fastPath?.fallback)
          bad(`${rel(entryFile)}: WebLLM fast-path eligibility and fallback are required`);
        const totalBytes = artifactBytes(manifest);
        if (row.artifactBytes !== totalBytes)
          bad(`catalog.json: ${famId}/${row.id}.artifactBytes disagrees with artifact manifest`);
      }

      // measurements
      if (entry.measurements) {
        const mPath = path.join(entryDir, entry.measurements);
        if (!fs.existsSync(mPath)) bad(`${rel(entryFile)}: measurements -> missing ${entry.measurements}`);
        else {
          const m = readJson(mPath);
          for (const e of v.validate("measurements.schema.json", m, rel(mPath))) bad(e);
          if (m.entry !== entry.id || m.family !== entry.family) bad(`${rel(mPath)}: entry/family do not match entry.json`);
          const measured = m.rows.filter((r) => r.kind === "measured");
          if (row.measuredHosts !== undefined && row.measuredHosts !== new Set(measured.map((r) => r.host.chip ?? r.host.browser)).size)
            bad(`catalog.json: ${famId}/${row.id}.measuredHosts disagrees with measurements.json`);
          for (const measuredRow of measured) {
            if (measuredRow.pairedSpeedup && measuredRow.protocol !== "interleaved-ab")
              bad(`${rel(mPath)}: pairedSpeedup requires protocol "interleaved-ab"`);
          }
          note(rel(mPath));
        }
      }
      if (entry.provenance && !fs.existsSync(path.join(entryDir, entry.provenance)))
        bad(`${rel(entryFile)}: provenance -> missing ${entry.provenance}`);
      if (entry.verification && !fs.existsSync(path.join(entryDir, entry.verification)))
        bad(`${rel(entryFile)}: verification -> missing ${entry.verification}`);
      note(rel(entryFile));
    }
  }

  return { errors, checked };
}

// ---------------------------------------------------------------------------
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const argv = process.argv.slice(2);
  const quiet = argv.includes("--quiet");
  const i = argv.indexOf("--entry");
  const only = i >= 0 ? argv[i + 1] : null;
  console.log(`webnn-catalog validate${only ? ` (${only})` : ""}`);
  const { errors, checked } = validateCatalog({ only, quiet });
  console.log(`\n${checked.length} files checked`);
  if (errors.length) {
    console.log(`\n${errors.length} problem${errors.length === 1 ? "" : "s"}:`);
    for (const e of errors) console.log(`  - ${e}`);
    console.log(`\nVALIDATE FAILED`);
    process.exitCode = 1;
  } else {
    console.log(`VALIDATE OK`);
  }
}
