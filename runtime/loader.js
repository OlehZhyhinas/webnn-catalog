// Catalog runtime: turn a catalog entry into built MLGraphs.
//
// A recipe is the exact MLGraphBuilder call sequence that built a graph once,
// recorded op by op with the operand shapes the backend itself inferred. This
// module replays it. It knows nothing about UNets, text encoders or diffusion;
// it knows the WebNN builder surface and the recipe schema.
//
//   const entry = await (await fetch(`${dir}/entry.json`)).json();
//   const rig   = await loadEntry(entry, "/weights", ctx, { baseUrl: dir });
//   const t     = await createEntryTensors(ctx, rig);
//   ctx.dispatch(rig.graphs.text.graph,  t.inputsFor("text"),  t.outputsFor("text"));
//
// ...or one graph at a time:
//
//   const source = constantSource(manifest.constants.image, { baseUrl });
//   const image  = await loadRecipe(recipeJson, source, context);
//   ctx.dispatch(image.graph, ins, outs);
//
// What this module does NOT do is choose an entry. Probing the machine,
// ranking entries and falling back are the consuming product's policy; the
// catalog has no opinion and ships no code for it. Hand it an entry and it
// gives you finished graphs.
//
// Everything else here is the plumbing that surrounds a graph on this backend:
// MLTensor allocation, the on-device chaining of one graph's output into the
// next graph's input, and the fingerprint assert that catches the silent CPU
// fallback before it costs an afternoon.
//
// Recipe schema (version 1):
//
//   { version, label, layout, inputs:  [{name, dataType, shape}],
//                             outputs: [{name, operand, dataType, shape}],
//                             constants: {k0: {dataType, shape, byteOffset, byteLength, tag?}},
//                             ops: [{id, type, inputs:[operandName], output, outputShape,
//                                    outputDataType, options, tag,
//                                    outputs?, outputShapes?}] }
//
// Operand names live in ONE namespace: graph inputs keep their own names,
// constants are k<N>, op results are v<N>. `constants` byteOffsets index the
// companion blob.

// ---------------------------------------------------------------------------
// Op call shape
// ---------------------------------------------------------------------------
//
// The recorder flattens a builder call's positional NON-operand arguments into
// the same `options` bag as the real options dictionary, so the recipe alone
// cannot tell `softmax(x, 2)` from `softmax(x, {axis: 2})`. This table is the
// recorder's table, inverted: for each method, which `options` keys are really
// positional and in what order. Replaying without it produces a graph that
// builds and computes the wrong thing (or, more often, throws deep inside
// Blink with a message about the wrong argument type).
//
// Keep in sync with POSITIONAL in the recorder that produced the recipe.
const POSITIONAL = {
  reshape: ["newShape"],
  expand: ["newShape"],
  softmax: ["axis"],
  cast: ["type"],
  concat: ["axis"],
  split: ["splits"],
  tile: ["repetitions"],
  pad: ["beginningPadding", "endingPadding"],
  argMin: ["axis"],
  argMax: ["axis"],
  slice: ["starts", "sizes"],
};

/** Builder methods whose first argument is an ARRAY of operands, not one. */
const VARIADIC = new Set(["concat"]);

/** Builder methods that return several operands. */
const MULTI_OUTPUT = new Set(["split"]);

const BYTES_PER_ELEMENT = {
  float32: 4,
  float16: 2,
  int32: 4,
  uint32: 4,
  int64: 8,
  uint64: 8,
  int8: 1,
  uint8: 1,
};

const VIEW_OF = {
  float32: Float32Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

const elementCount = (shape) => shape.reduce((a, b) => a * b, 1);

export function byteLengthOf(dataType, shape) {
  const w = BYTES_PER_ELEMENT[dataType];
  if (w === undefined) throw new Error(`unknown dataType "${dataType}"`);
  return elementCount(shape) * w;
}

/**
 * A typed view over raw bytes, for `builder.constant(descriptor, view)`.
 *
 * float16 has no universal view: Chrome 152 has a global Float16Array, but
 * builder.constant() also accepts the raw little-endian bytes as a Uint8Array,
 * which is what the recorder captured and what stays zero-copy. So float16 (and
 * anything else without a view constructor) is handed over as bytes.
 */
function constantView(bytes, dataType) {
  const Ctor = VIEW_OF[dataType];
  if (!Ctor) return bytes; // float16 and friends: raw little-endian bytes
  if (bytes.byteOffset % Ctor.BYTES_PER_ELEMENT !== 0) {
    // A misaligned window into the blob cannot be viewed in place. The recipe's
    // offsets are 8-byte aligned, so this is a corrupt-or-repacked blob.
    return new Ctor(bytes.slice().buffer);
  }
  return new Ctor(bytes.buffer, bytes.byteOffset, bytes.byteLength / Ctor.BYTES_PER_ELEMENT);
}

// ---------------------------------------------------------------------------
// Constant sources
// ---------------------------------------------------------------------------
//
// The image recipe's constants are 1.65 GiB. Three ways to get them in:
//
//   1. whole file      -- one ArrayBuffer, one fetch. Simplest, and what the
//                         workbench does (1.7 s on localhost). Peak JS heap is
//                         the whole blob, plus whatever Chrome copies into the
//                         graph.
//   2. chunk list      -- the manifest names byte ranges; each is fetched, its
//                         constants are built out of it, and it is released
//                         before the next one is fetched. Peak JS heap is one
//                         chunk. Constants that straddle a boundary are
//                         stitched across the seam.
//   3. custom fetcher  -- any object with `read(byteOffset, byteLength)`.
//
// (2) exists because a single 1.65 GiB ArrayBuffer is a hostile allocation on a
// machine that is also holding the same bytes inside the graph; it is the
// mechanism a published, HTTP-Range-served blob wants.

/** Whole-blob source: an ArrayBuffer already in hand. */
export function bufferSource(arrayBuffer) {
  const all = new Uint8Array(arrayBuffer);
  return {
    kind: "buffer",
    totalBytes: all.byteLength,
    ranges: [{ byteOffset: 0, byteLength: all.byteLength }],
    async fetchRange(byteOffset, byteLength) {
      return all.subarray(byteOffset, byteOffset + byteLength);
    },
  };
}

/**
 * HTTP source. With no `chunks` this is one request for the whole file; with
 * `chunks` (a list of {byteOffset, byteLength}) it is one ranged request each,
 * issued only when that chunk's constants are about to be built.
 *
 * The server must answer Range requests for the chunked path. `serve.mjs` in
 * this repo does; so does every static host worth publishing to.
 */
export function httpSource(url, { chunks = null, totalBytes = null, fetchImpl = fetch } = {}) {
  const ranges =
    chunks && chunks.length
      ? chunks.map((c) => ({ byteOffset: c.byteOffset, byteLength: c.byteLength }))
      : [{ byteOffset: 0, byteLength: totalBytes ?? Infinity }];
  return {
    kind: chunks && chunks.length ? "http-chunked" : "http-whole",
    totalBytes,
    ranges,
    async fetchRange(byteOffset, byteLength) {
      const headers =
        byteOffset === 0 && (byteLength === totalBytes || byteLength === Infinity)
          ? {}
          : { Range: `bytes=${byteOffset}-${byteOffset + byteLength - 1}` };
      const res = await fetchImpl(url, { headers });
      if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
      if (Object.keys(headers).length && res.status !== 206)
        throw new Error(`${url} ignored the Range header (HTTP ${res.status}); chunked load needs a Range-capable server`);
      return new Uint8Array(await res.arrayBuffer());
    },
  };
}

/**
 * Build a source from a catalog manifest entry.
 *
 * `entry` is one of manifest.json's `constants` records: {file, bytes, sha256,
 * url, chunks?}. `baseUrl` is where `file` lives when `url` is still null (the
 * pre-publication case: weights on disk, served locally).
 */
export function constantSource(entry, { baseUrl = "", fetchImpl = fetch } = {}) {
  const url = entry.url ?? `${baseUrl.replace(/\/$/, "")}/${entry.file}`;
  return httpSource(url, { chunks: entry.chunks ?? null, totalBytes: entry.bytes, fetchImpl });
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

/**
 * Create every constant in the recipe, walking the source's ranges so that at
 * most one range is resident at a time.
 *
 * Constants are created in byteOffset order regardless of the order they are
 * used in, which is what lets a chunked source release each chunk as it goes.
 */
async function buildConstants(builder, recipe, source, onProgress) {
  const names = Object.keys(recipe.constants);
  names.sort((a, b) => recipe.constants[a].byteOffset - recipe.constants[b].byteOffset);

  let end = 0;
  for (const n of names) {
    const c = recipe.constants[n];
    if (byteLengthOf(c.dataType, c.shape) !== c.byteLength)
      throw new Error(`constant ${n}: shape ${JSON.stringify(c.shape)} of ${c.dataType} is not ${c.byteLength} bytes`);
    end = Math.max(end, c.byteOffset + c.byteLength);
  }

  const operands = new Map();
  const ranges = source.ranges.length
    ? source.ranges
    : [{ byteOffset: 0, byteLength: end }];

  let i = 0; // next constant to place
  let carry = null; // {name, bytes: Uint8Array, filled: number} straddling a seam
  let bytesRead = 0;

  for (const r of ranges) {
    const length = Number.isFinite(r.byteLength) ? r.byteLength : end - r.byteOffset;
    const chunk = await source.fetchRange(r.byteOffset, length);
    bytesRead += chunk.byteLength;
    const base = r.byteOffset;

    if (carry) {
      const need = carry.bytes.byteLength - carry.filled;
      const take = Math.min(need, chunk.byteLength);
      carry.bytes.set(chunk.subarray(0, take), carry.filled);
      carry.filled += take;
      if (carry.filled === carry.bytes.byteLength) {
        const c = recipe.constants[carry.name];
        operands.set(carry.name, builder.constant({ dataType: c.dataType, shape: c.shape }, constantView(carry.bytes, c.dataType)));
        carry = null;
      } else {
        continue; // this whole chunk was swallowed by one enormous constant
      }
    }

    while (i < names.length) {
      const name = names[i];
      const c = recipe.constants[name];
      if (c.byteOffset >= base + chunk.byteLength) break;
      const start = c.byteOffset - base;
      if (start < 0) throw new Error(`constant ${name} at ${c.byteOffset} is before the current range at ${base}`);
      if (start + c.byteLength <= chunk.byteLength) {
        const bytes = chunk.subarray(start, start + c.byteLength);
        operands.set(name, builder.constant({ dataType: c.dataType, shape: c.shape }, constantView(bytes, c.dataType)));
        i++;
      } else {
        // straddles the end of this range: copy what is here and wait.
        carry = { name, bytes: new Uint8Array(c.byteLength), filled: chunk.byteLength - start };
        carry.bytes.set(chunk.subarray(start), 0);
        i++;
        break;
      }
    }
    onProgress?.({ phase: "constants", bytesRead, totalBytes: end, built: operands.size, total: names.length });
  }

  if (carry) throw new Error(`constant ${carry.name} was never completed: the ranges do not cover the blob`);
  if (operands.size !== names.length)
    throw new Error(`built ${operands.size} of ${names.length} constants: the ranges do not cover the blob`);
  return operands;
}

/**
 * Turn one recorded op into a builder call.
 *
 * The recorded `inputs` array is [...positional operands, ...operand-valued
 * options], in that order, because the recorder walked the argument list and
 * an options bag is always last. So the operand-valued options can be sheared
 * off the tail by count, and what is left is positional.
 */
function callOp(builder, op, operands) {
  const resolve = (n) => {
    const o = operands.get(n);
    if (o === undefined) throw new Error(`op ${op.id} (${op.type}) uses undefined operand "${n}"`);
    return o;
  };

  const options = {};
  const optionOperandNames = [];
  for (const [k, v] of Object.entries(op.options ?? {})) {
    if (typeof v === "string" && operands.has(v)) {
      options[k] = resolve(v);
      optionOperandNames.push(v);
    } else {
      options[k] = v;
    }
  }

  const positionalNames = op.inputs.slice(0, op.inputs.length - optionOperandNames.length);
  const tail = op.inputs.slice(op.inputs.length - optionOperandNames.length);
  // Cheap integrity check on the assumption above.
  if (tail.slice().sort().join("|") !== optionOperandNames.slice().sort().join("|"))
    throw new Error(`op ${op.id} (${op.type}): operand-valued options ${optionOperandNames} are not the tail of inputs ${op.inputs}`);

  const args = VARIADIC.has(op.type)
    ? [positionalNames.map(resolve)]
    : positionalNames.map(resolve);

  // Positional non-operand arguments, in the recorder's order, removed from the
  // options bag so they are not also passed as a dictionary key.
  const spec = POSITIONAL[op.type] ?? [];
  const extra = [];
  for (const key of spec) {
    extra.push(options[key]);
    delete options[key];
  }
  while (extra.length && extra[extra.length - 1] === undefined) extra.pop();
  args.push(...extra);

  if (Object.keys(options).length) args.push(options);

  const fn = builder[op.type];
  if (typeof fn !== "function")
    throw new Error(`op ${op.id}: MLGraphBuilder has no method "${op.type}" in this browser`);
  return fn.apply(builder, args);
}

/**
 * Replay a recipe into a built MLGraph.
 *
 * @param {object} recipe   parsed recipe JSON
 * @param {object|ArrayBuffer} source  a constant source, or an ArrayBuffer
 * @param {MLContext} context
 * @param {object} [opts]   {onProgress, builder}
 * @returns {{graph, inputs, outputs, label, layout, stats}}
 *   `inputs`  : {name: {dataType, shape}}   -- what dispatch() must bind
 *   `outputs` : {name: {dataType, shape}}   -- ditto
 */
export async function loadRecipe(recipe, source, context, { onProgress, builder: given = null } = {}) {
  if (recipe.version !== 1) throw new Error(`recipe version ${recipe.version} is not supported (expected 1)`);
  const src = source instanceof ArrayBuffer ? bufferSource(source) : source;
  const builder = given ?? new MLGraphBuilder(context);

  const t0 = now();
  const operands = await buildConstants(builder, recipe, src, onProgress);
  const tConstants = now() - t0;

  for (const spec of recipe.inputs) {
    if (operands.has(spec.name)) throw new Error(`input "${spec.name}" collides with a constant name`);
    operands.set(spec.name, builder.input(spec.name, { dataType: spec.dataType, shape: spec.shape }));
  }

  const t1 = now();
  const seen = new Set();
  for (const op of recipe.ops) {
    const result = callOp(builder, op, operands);
    const outNames = op.outputs ?? [op.output];
    const results = MULTI_OUTPUT.has(op.type) || Array.isArray(result) ? result : [result];
    if (results.length !== outNames.length)
      throw new Error(`op ${op.id} (${op.type}) returned ${results.length} operands, recipe names ${outNames.length}`);
    for (let k = 0; k < outNames.length; k++) {
      // An aliased operand (the builder handed back something already named)
      // keeps its first name, exactly as the recorder did.
      if (!operands.has(outNames[k])) operands.set(outNames[k], results[k]);
      seen.add(outNames[k]);
    }
    onProgress?.({ phase: "ops", done: op.id + 1, total: recipe.ops.length });
  }
  const tEmit = now() - t1;

  const outputOperands = {};
  for (const o of recipe.outputs) {
    const op = operands.get(o.operand);
    if (op === undefined) throw new Error(`output "${o.name}" names undefined operand "${o.operand}"`);
    outputOperands[o.name] = op;
  }

  const t2 = now();
  const graph = await builder.build(outputOperands);
  const tBuild = now() - t2;

  const inputs = {};
  for (const s of recipe.inputs) inputs[s.name] = { dataType: s.dataType, shape: s.shape };
  const outputs = {};
  for (const s of recipe.outputs) outputs[s.name] = { dataType: s.dataType, shape: s.shape };

  return {
    graph,
    inputs,
    outputs,
    label: recipe.label,
    layout: recipe.layout,
    stats: {
      ops: recipe.ops.length,
      constants: Object.keys(recipe.constants).length,
      constantBytes: src.totalBytes ?? null,
      operands: operands.size,
      opResults: seen.size,
      constantsMs: +tConstants.toFixed(1),
      emitMs: +tEmit.toFixed(1),
      buildMs: +tBuild.toFixed(1),
    },
  };
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

// ---------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------
//
// An entry is a whole configuration: several graphs, the manifest naming their
// constants, and the on-device links between them. loadEntry() fetches what it
// needs relative to the entry directory and builds every graph.

const joinUrl = (base, file) => (/^([a-z]+:)?\/\//i.test(file) ? file : `${String(base).replace(/\/$/, "")}/${file}`);

/**
 * Build every graph an entry declares.
 *
 * @param {object} entry    parsed entry.json
 * @param {string|function|object|null} constants  where the constants come from:
 *        a base URL for the blobs, a `(name, manifestRecord) => source` factory,
 *        a `{name: source}` map, or null to use each manifest record's own `url`.
 * @param {MLContext} context
 * @param {object} [opts]  {baseUrl, fetchImpl, onProgress, only, manifest, recipes, order}
 * @returns {{entry, manifest, graphs, chain, inputs, outputs, stats}}
 *
 * `graphs[name]` is exactly what loadRecipe() returns. Graphs are built in the
 * order given, largest constants first by default: building the 1650 MiB graph
 * before the 649 MiB one keeps peak resident bytes lower.
 */
export async function loadEntry(entry, constants = null, context, opts = {}) {
  const {
    baseUrl = entry.baseUrl ?? ".",
    fetchImpl = fetch,
    onProgress,
    only = null,
    manifest: givenManifest = null,
    recipes: givenRecipes = null,
    order = null,
  } = opts;

  const j = async (url) => {
    const res = await fetchImpl(url);
    if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
    return res.json();
  };

  const manifest = givenManifest ?? (await j(joinUrl(baseUrl, entry.constants)));
  const names = Object.keys(entry.graphs).filter((k) => k !== "chain" && (!only || only.includes(k)));

  const sourceFor = (graphName) => {
    const key = entry.graphs[graphName].constants;
    const record = manifest.constants[key];
    if (!record) throw new Error(`entry ${entry.id}: graph "${graphName}" names constants "${key}", which the manifest does not have`);
    if (typeof constants === "function") return constants(key, record);
    if (constants && typeof constants === "object") {
      const s = constants[key];
      if (!s) throw new Error(`no constant source supplied for "${key}"`);
      return s;
    }
    if (typeof constants === "string") return constantSource(record, { baseUrl: constants, fetchImpl });
    if (record.url == null)
      throw new Error(`constants "${key}" are unpublished (manifest url is null): pass a base URL or a source for them`);
    return constantSource(record, { fetchImpl });
  };

  const build = order ?? [...names].sort(
    (a, b) => (manifest.constants[entry.graphs[b].constants]?.bytes ?? 0) - (manifest.constants[entry.graphs[a].constants]?.bytes ?? 0),
  );

  const graphs = {};
  for (const name of build) {
    const spec = entry.graphs[name];
    const recipe = givenRecipes?.[name] ?? (await j(joinUrl(baseUrl, spec.recipe)));
    graphs[name] = await loadRecipe(recipe, sourceFor(name), context, {
      onProgress: onProgress ? (p) => onProgress({ ...p, graph: name }) : undefined,
    });
  }

  const chain = (entry.graphs.chain ?? []).map(([from, to]) => {
    const [fg, fo] = from.split(".");
    const [tg, ti] = to.split(".");
    return { from: { graph: fg, name: fo }, to: { graph: tg, name: ti } };
  });

  const inputs = {}, outputs = {};
  for (const name of Object.keys(graphs)) {
    inputs[name] = graphs[name].inputs;
    outputs[name] = graphs[name].outputs;
  }

  return {
    entry,
    manifest,
    graphs,
    chain,
    inputs,
    outputs,
    stats: Object.fromEntries(Object.entries(graphs).map(([k, g]) => [k, g.stats])),
  };
}

/**
 * Allocate every tensor a loaded entry needs, sharing one MLTensor across each
 * chain link so the intermediate never crosses into JS.
 *
 * `readable` comes from the entry: an output the entry marks readable is
 * allocated readable, and so is any output that feeds a chain link, because
 * readTensor() on it is the cheapest completion fence WebNN offers. Everything
 * else is allocated non-readable and still bound, because WebNN requires every
 * declared output to be bound at dispatch.
 */
export async function createEntryTensors(context, rig, { readable = null, writable = null } = {}) {
  const key = (g, n) => `${g}.${n}`;
  const live = (rig.chain ?? []).filter((l) => rig.graphs[l.from.graph] && rig.graphs[l.to.graph]);
  const feeds = new Set(live.map((l) => key(l.from.graph, l.from.name)));
  const fedBy = new Map(live.map((l) => [key(l.to.graph, l.to.name), key(l.from.graph, l.from.name)]));
  const declared = (g, n) => (rig.entry.graphs[g]?.outputs ?? []).find((o) => o.name === n);

  const tensors = {};
  for (const [g, loaded] of Object.entries(rig.graphs)) {
    for (const [n, spec] of Object.entries(loaded.outputs)) {
      const k = key(g, n);
      const want = readable?.includes(k) ?? (feeds.has(k) || declared(g, n)?.readable !== false);
      tensors[k] = await context.createTensor({ dataType: spec.dataType, shape: spec.shape, readable: want });
    }
  }
  for (const [g, loaded] of Object.entries(rig.graphs)) {
    for (const [n, spec] of Object.entries(loaded.inputs)) {
      const k = key(g, n);
      const from = fedBy.get(k);
      if (from) {
        const producer = tensors[from];
        if (!producer) throw new Error(`chain ${from} -> ${k}: the producer graph is not loaded`);
        tensors[k] = producer; // ONE tensor, two roles
        continue;
      }
      tensors[k] = await context.createTensor({
        dataType: spec.dataType,
        shape: spec.shape,
        writable: writable?.includes(k) ?? true,
      });
    }
  }

  const pick = (g, which) =>
    Object.fromEntries(Object.keys(rig.graphs[g][which]).map((n) => [n, tensors[key(g, n)]]));
  return {
    tensors,
    inputsFor: (g) => pick(g, "inputs"),
    outputsFor: (g) => pick(g, "outputs"),
    get: (g, n) => tensors[key(g, n)],
  };
}

/**
 * Which builder methods a recipe needs. Useful before loading 1.6 GB to find
 * out the browser is missing an op.
 */
export function opCoverage(recipe) {
  const histo = {};
  for (const op of recipe.ops) histo[op.type] = (histo[op.type] ?? 0) + 1;
  return histo;
}

export function checkOpSupport(recipe, builderOrContext) {
  const builder =
    builderOrContext instanceof MLGraphBuilder ? builderOrContext : new MLGraphBuilder(builderOrContext);
  const missing = Object.keys(opCoverage(recipe)).filter((t) => typeof builder[t] !== "function");
  return { missing, ok: missing.length === 0 };
}

// ---------------------------------------------------------------------------
// Backend fingerprint
// ---------------------------------------------------------------------------

/**
 * Assert that this MLContext is the Core ML backend and not the CPU fallback.
 *
 * On macOS, Chromium gates the Core ML backend on `!is_incognito`. An
 * off-the-record profile -- which is what Playwright's `chromium.launch()` and
 * Chrome's own incognito window both give you -- silently lands on
 * TFLite/XNNPACK on the CPU instead. Nothing fails; the graph builds in
 * milliseconds instead of seconds and then runs about 50x slower. The two
 * observable tells:
 *
 *   preferredInputLayout   "nchw" on Core ML, "nhwc" on TFLite
 *   rankRange.max          5 on Core ML, 8 on TFLite
 *
 * Call this immediately after createContext(), before spending a minute
 * building a graph you are about to mis-measure.
 */
export function assertCoreMLFingerprint(context, { throwOnMismatch = true } = {}) {
  let limits;
  try {
    limits = context.opSupportLimits();
  } catch (e) {
    throw new Error(`opSupportLimits() failed: ${e}. WebNN needs Chrome with --enable-features=WebMachineLearningNeuralNetwork,WebNNCoreML`);
  }
  const preferredInputLayout = limits.preferredInputLayout;
  const maxRank = limits.add?.a?.rankRange?.max ?? null;
  const ok = preferredInputLayout === "nchw" && maxRank === 5;
  const report = { ok, preferredInputLayout, maxRank, backend: ok ? "coreml" : "not-coreml" };
  if (!ok && throwOnMismatch) {
    throw new Error(
      `Not the Core ML backend: preferredInputLayout="${preferredInputLayout}" (expected "nchw"), ` +
        `add rankRange.max=${maxRank} (expected 5). This is the TFLite/XNNPACK CPU fallback, which is ` +
        `roughly 50x slower and will build in milliseconds instead of seconds.\n` +
        `Chromium gates the Core ML backend on a NON-incognito profile. Launch Chrome with a real ` +
        `on-disk user-data-dir: Playwright's chromium.launchPersistentContext(profileDir, {channel:"chrome"}), ` +
        `never chromium.launch(). Required flags: --enable-unsafe-webgpu ` +
        `--enable-features=WebMachineLearningNeuralNetwork,WebMachineLearningNeuralNetworkExperimentalFeatures,WebNNCoreML,WebGPUExperimentalFeatures ` +
        `--enable-dawn-features=allow_unsafe_apis`,
    );
  }
  return report;
}

// ---------------------------------------------------------------------------
// MLTensor helpers
// ---------------------------------------------------------------------------
//
// A graph's inputs and outputs are MLTensors, which live on the device. The
// chaining pattern that matters for a two-graph pipeline: the tensor that
// receives graph A's output IS the tensor bound to graph B's input, so the
// [1,77,1024] embedding never crosses back into JS. See chainTensor().

/** Allocate the input tensors a loaded graph declares. */
export async function createInputTensors(context, loaded, { writable = true, only = null } = {}) {
  const out = {};
  for (const [name, spec] of Object.entries(loaded.inputs)) {
    if (only && !only.includes(name)) continue;
    out[name] = await context.createTensor({ dataType: spec.dataType, shape: spec.shape, writable });
  }
  return out;
}

/**
 * Allocate the output tensors a loaded graph declares.
 *
 * `readable` names the outputs JS will actually read back; everything else is
 * allocated non-readable. WebNN requires EVERY declared graph output to be
 * bound at dispatch, including ones you do not want -- a recipe recorded with a
 * debug output still carries it.
 */
export async function createOutputTensors(context, loaded, { readable = [] } = {}) {
  const want = new Set(readable);
  const out = {};
  for (const [name, spec] of Object.entries(loaded.outputs)) {
    out[name] = await context.createTensor({
      dataType: spec.dataType,
      shape: spec.shape,
      readable: want.has(name),
    });
  }
  return out;
}

/**
 * One MLTensor that is graph A's output and graph B's input, so the value never
 * leaves the device.
 *
 * Both graphs must be on the SAME MLContext. When they are not (two device
 * types, say), pass `bridgeVia: "host"` and the value is read back and written
 * across, which is what the extra ~0.1 ms buys you.
 */
export async function chainTensor(context, producer, producerOutput, consumer, consumerInput, { readable = true } = {}) {
  const a = producer.outputs[producerOutput];
  const b = consumer.inputs[consumerInput];
  if (!a) throw new Error(`"${producerOutput}" is not an output of ${producer.label}`);
  if (!b) throw new Error(`"${consumerInput}" is not an input of ${consumer.label}`);
  if (a.dataType !== b.dataType || String(a.shape) !== String(b.shape))
    throw new Error(
      `cannot chain ${producer.label}.${producerOutput} ${a.dataType}[${a.shape}] into ` +
        `${consumer.label}.${consumerInput} ${b.dataType}[${b.shape}]`,
    );
  // `readable` so readTensor() on it can serve as the completion fence for the
  // producer without a second allocation.
  return context.createTensor({ dataType: a.dataType, shape: a.shape, readable });
}

/**
 * Write host data into a tensor. Accepts a typed array or an ArrayBuffer.
 */
export function writeTensor(context, tensor, data) {
  context.writeTensor(tensor, data);
}

/**
 * Read a tensor back, optionally into a buffer you own.
 *
 * The two-argument overload (Chrome 152, ml_context.idl) fills a view you
 * allocated once instead of handing back a fresh ArrayBuffer per frame. For a
 * 1 MB image at 15 fps that is the difference between zero and a megabyte of
 * garbage per frame.
 */
export function readTensor(context, tensor, into = null) {
  return into ? context.readTensor(tensor, into) : context.readTensor(tensor);
}

/**
 * readTensor() as a completion fence: WebNN dispatch() is queued, and reading
 * any output of the last graph in the queue resolves only once it has run.
 */
export async function fence(context, tensor, into = null) {
  await readTensor(context, tensor, into);
}

/** Dispatch a loaded graph. Thin, but it keeps call sites symmetrical. */
export function dispatch(context, loaded, inputs, outputs) {
  context.dispatch(loaded.graph, inputs, outputs);
}

// ---------------------------------------------------------------------------
// Autoregressive decoding
// ---------------------------------------------------------------------------

/**
 * Run a decode graph that computes k greedy steps per dispatch over static
 * caches, until it emits `eos` or `maxNewTokens` tokens. Everything the loop
 * needs is data: `spec` is a family's `contract.chaining.autoregressive`
 * block, and the tensors come from createEntryTensors(), which already
 * allocated one tensor per cache input and one per cache output; those two
 * are the two sets the caches ping-pong between (an MLTensor cannot be both an
 * input and an output of one dispatch).
 *
 *   spec = {graph, tokenInput, positionInput, tokensOutput,
 *           caches: [[inputName, outputName], ...], zeroCachesPerSequence,
 *           bos, eos, maxNewTokens}
 *
 * The unroll factor k is the tokens output's length; the cache length is the
 * cache tensors' shape. Returns {tokens, dispatches, endedWithEos}. The
 * graph's other inputs (the encoder K/V a chain link filled) are bound as
 * createEntryTensors() left them.
 */
export async function autoregressive(context, rig, tensors, spec, { maxNewTokens = null, tokensView = null } = {}) {
  const g = spec.graph;
  const loaded = rig.graphs[g];
  if (!loaded) throw new Error(`autoregressive: graph "${g}" is not loaded`);
  const k = loaded.outputs[spec.tokensOutput].shape.reduce((a, b) => a * b, 1);
  const cacheLen = Math.min(...spec.caches.map(([i]) => Math.max(...loaded.inputs[i].shape)));
  const limit = Math.min(maxNewTokens ?? spec.maxNewTokens ?? cacheLen, cacheLen);
  const sets = [
    spec.caches.map(([i]) => tensors.get(g, i)),
    spec.caches.map(([, o]) => tensors.get(g, o)),
  ];
  if (spec.zeroCachesPerSequence) {
    // the first dispatch of a sequence must see clean caches: the update adds
    // into the row at `step`, it does not replace it
    for (const [i] of spec.caches) {
      const t = tensors.get(g, i);
      const n = loaded.inputs[i].shape.reduce((a, b) => a * b, 1);
      context.writeTensor(t, new Uint8Array(n * byteLengthOf(loaded.inputs[i].dataType, [1])));
    }
  }
  const tokT = tensors.get(g, spec.tokenInput), stepT = tensors.get(g, spec.positionInput);
  const tokBuf = new Int32Array([spec.bos ?? 0]), stepBuf = new Int32Array([0]);
  const out = tokensView ?? new Int32Array(k);
  const baseIn = tensors.inputsFor(g), baseOut = tensors.outputsFor(g);
  const tokens = [];
  let cur = 0, dispatches = 0, ended = false;
  while (tokens.length < limit) {
    context.writeTensor(tokT, tokBuf);
    context.writeTensor(stepT, stepBuf);
    const inputs = { ...baseIn }, outputs = { ...baseOut };
    spec.caches.forEach(([i, o], idx) => { inputs[i] = sets[cur][idx]; outputs[o] = sets[cur ^ 1][idx]; });
    context.dispatch(loaded.graph, inputs, outputs);
    cur ^= 1;
    dispatches++;
    await context.readTensor(tensors.get(g, spec.tokensOutput), out);
    for (let j = 0; j < k && tokens.length < limit; j++) {
      tokens.push(out[j]);
      if (out[j] === spec.eos) { ended = true; break; }
    }
    if (ended) break;
    tokBuf[0] = tokens[tokens.length - 1];
    stepBuf[0] = tokens.length;
  }
  return { tokens, dispatches, endedWithEos: ended, k, cacheLen };
}
