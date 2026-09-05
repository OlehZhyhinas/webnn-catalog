// The page half of scripts/verify.mjs. Everything that touches WebNN happens
// here; the Node driver only launches Chrome, serves files and prints a table.
//
// It is driven by an ENTRY: the graphs, their chaining and which outputs are
// read back all come out of entry.json. The reference case, the bars and the
// checks are the sd-turbo-512-1step family's, and this file says so rather than
// pretending to be generic.

import {
  loadEntry,
  createEntryTensors,
  constantSource,
  assertCoreMLFingerprint,
  opCoverage,
  checkOpSupport,
  autoregressive,
} from "../runtime/loader.js";

const KNOWN_FAMILY = "sd-turbo-512-1step";

const logEl = document.getElementById("log");
const lines = [];
const log = (s) => {
  lines.push(s);
  logEl.textContent = lines.join("\n");
  console.log(`[verify] ${s}`);
};

const j = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); });
const b = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.arrayBuffer(); });

const median = (a) => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const stats = (a) => ({
  medianMs: +median(a).toFixed(2),
  minMs: +Math.min(...a).toFixed(2),
  maxMs: +Math.max(...a).toFixed(2),
  meanMs: +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(2),
  n: a.length,
});

/** NCHW -> NHWC on raw float16 bytes, as a 16-bit element permutation. */
function nchwToNhwc16(buf, N, C, H, W) {
  const src = new Uint16Array(buf);
  const dst = new Uint16Array(src.length);
  const HW = H * W;
  for (let n = 0; n < N; n++) {
    const so = n * C * HW, dof = n * HW * C;
    for (let c = 0; c < C; c++) for (let p = 0; p < HW; p++) dst[dof + p * C + c] = src[so + c * HW + p];
  }
  return dst;
}

async function sha256Hex(buffer) {
  const d = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(d)].map((x) => x.toString(16).padStart(2, "0")).join("");
}

/** Decode a PNG to RGBA bytes. */
async function pngRGBA(url) {
  const bmp = await createImageBitmap(await (await fetch(url)).blob());
  const cv = new OffscreenCanvas(bmp.width, bmp.height);
  const cx = cv.getContext("2d", { willReadFrequently: true });
  cx.drawImage(bmp, 0, 0);
  const d = cx.getImageData(0, 0, bmp.width, bmp.height);
  return { data: d.data, width: bmp.width, height: bmp.height };
}

/** PSNR over the RGB channels of two RGBA byte arrays, plus the max byte diff. */
function psnrRGB(a, bb) {
  if (a.length !== bb.length) return { psnrDb: null, error: `size ${a.length} vs ${bb.length}` };
  let se = 0, n = 0, maxDiff = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let k = 0; k < 3; k++) {
      const d = a[i + k] - bb[i + k];
      se += d * d; n++;
      if (Math.abs(d) > maxDiff) maxDiff = Math.abs(d);
    }
  }
  const mse = se / n;
  return { psnrDb: mse === 0 ? Infinity : +(10 * Math.log10((255 * 255) / mse)).toFixed(3), maxByteDiff: maxDiff, exact: mse === 0 };
}

export async function run(opts = {}) {
  const { runs = 20, warmup = 5, weightsBase = "/weights", chunkMB = 0, prompt = null, entryPath } = opts;
  if (!entryPath) throw new Error("no entryPath: verify.mjs must name the entry to run");
  const R = { opts: { runs, warmup, weightsBase, chunkMB, entryPath }, errors: [], checks: {}, timings: {} };
  const T = () => performance.now();

  // ---- the entry ----------------------------------------------------------
  const entry = await j(`${entryPath}/entry.json`);
  R.entry = { id: entry.id, family: entry.family, variant: entry.variant };
  // A verification set that declares kind "tokens" is checked by the generic
  // autoregressive path below (any family whose contract carries a
  // chaining.autoregressive block). Everything else is the sd-turbo reference
  // case, which this file spells out by hand.
  {
    const expected0 = await j(`${entryPath}/verification/expected.json`);
    if (expected0.kind === "tokens") return runTokens({ R, entry, entryPath, expected: expected0, runs, warmup, weightsBase, chunkMB });
  }
  if (entry.family !== KNOWN_FAMILY)
    throw new Error(`this harness knows the ${KNOWN_FAMILY} family's reference case only; the entry is from ${entry.family}`);
  const familyBase = `/families/${entry.family}`;
  const { default: ClipTokenizer } = await import(`${familyBase}/tokenizer.js`);

  const [manifest, expected, refIds] = await Promise.all([
    j(`${entryPath}/${entry.constants}`),
    j(`${entryPath}/verification/expected.json`),
    j(`${entryPath}/verification/input_ids.json`),
  ]);
  const graphNames = Object.keys(entry.graphs).filter((k) => k !== "chain");
  const recipes = Object.fromEntries(await Promise.all(
    graphNames.map(async (g) => [g, await j(`${entryPath}/${entry.graphs[g].recipe}`)]),
  ));

  // ---- context + fingerprint ---------------------------------------------
  if (!navigator.ml) throw new Error("navigator.ml is missing: WebNN is not enabled in this browser");
  let t = T();
  const ctx = await navigator.ml.createContext({ deviceType: entry.target.backend.deviceType ?? "gpu" });
  R.timings.createContextMs = +(T() - t).toFixed(1);
  R.fingerprint = assertCoreMLFingerprint(ctx);
  log(`context: ${R.fingerprint.backend}, preferredInputLayout=${R.fingerprint.preferredInputLayout}, rank max ${R.fingerprint.maxRank}`);
  log(`entry: ${entry.family}/${entry.id} (variant ${entry.variant}), built for ${entry.target.backend.name} on ${entry.target.host.chip ?? entry.target.gpu.vendor}`);
  ctx.lost?.then((i) => { R.contextLost = String(i?.message ?? i); log(`CONTEXT LOST: ${R.contextLost}`); }).catch(() => {});

  // ---- op support ---------------------------------------------------------
  R.opCoverage = Object.fromEntries(graphNames.map((g) => [g, opCoverage(recipes[g])]));
  R.checks.opSupport = Object.fromEntries(graphNames.map((g) => [g, checkOpSupport(recipes[g], ctx)]));
  const missing = graphNames.flatMap((g) => R.checks.opSupport[g].missing);
  if (missing.length) throw new Error(`MLGraphBuilder is missing: ${[...new Set(missing)].join(", ")}`);
  log(graphNames.map((g) => `${g} ${Object.keys(R.opCoverage[g]).length} types / ${recipes[g].ops.length} ops`).join(", ") + " - all supported");

  // ---- constant sources ---------------------------------------------------
  // With --chunk-mb the blob is fetched in ranges and each one is released
  // before the next, which is the path a published, Range-served blob wants.
  // The chunk boundaries here are deliberately NOT constant-aligned, so the
  // loader's straddle-stitching is exercised rather than skipped.
  const mkChunks = (bytes) => {
    if (!chunkMB) return null;
    const size = chunkMB * 1024 * 1024 + 1; // +1: never land on an 8-byte boundary
    const out = [];
    for (let o = 0; o < bytes; o += size) out.push({ byteOffset: o, byteLength: Math.min(size, bytes - o) });
    return out;
  };
  const sources = Object.fromEntries(Object.entries(manifest.constants).map(([key, c]) =>
    [key, constantSource({ ...c, chunks: mkChunks(c.bytes) }, { baseUrl: weightsBase })]));

  // ---- build every graph the entry declares -------------------------------
  // loadEntry builds the largest constants first, which keeps peak resident
  // bytes lower than building the small graph first would.
  t = T();
  const rig = await loadEntry(entry, sources, ctx, {
    baseUrl: entryPath,
    recipes,
    onProgress: (p) => {
      if (p.phase === "constants")
        log(`  ${p.graph} constants ${(p.bytesRead / 1e6).toFixed(0)}/${(p.totalBytes / 1e6).toFixed(0)} MB, ${p.built}/${p.total}`);
    },
  });
  R.timings.loadEntryMs = +(T() - t).toFixed(1);
  R.imageStats = rig.graphs.image.stats;
  R.textStats = rig.graphs.text.stats;
  for (const g of graphNames) log(`${g} graph built: ${JSON.stringify(rig.graphs[g].stats)}`);

  // ---- tokenizer ----------------------------------------------------------
  const tok = await ClipTokenizer.load(`${familyBase}/tokenizer/`);
  const usePrompt = prompt ?? refIds.prompt;
  const enc = tok.encode(usePrompt, 77);
  const idsMatch = usePrompt === refIds.prompt && refIds.ids.every((v, i) => enc.ids[i] === v);
  R.checks.tokenizer = { prompt: usePrompt, matchesReference: idsMatch, nTokens: enc.nTokens, stats: tok.stats() };
  if (usePrompt === refIds.prompt && !idsMatch) R.errors.push("tokenizer output does not match verification/input_ids.json");
  log(`tokenizer: ${enc.nTokens} tokens, matches reference ids: ${idsMatch}`);

  // ---- tensors ------------------------------------------------------------
  // One tensor is the text graph's `out` AND the image graph's
  // `encoder_hidden_states`, because entry.graphs.chain says so: the embedding
  // never crosses into JS. Every declared output is allocated and bound,
  // including the debug `latent`, which entry.json marks non-readable.
  const tensors = await createEntryTensors(ctx, rig);
  const inputIds = tensors.get("text", "input_ids");
  ctx.writeTensor(inputIds, enc.ids);
  const embedding = tensors.get("text", "out");

  const [rawBuf, scaledBuf, refEmbBuf] = await Promise.all([
    b(`${entryPath}/verification/latent_raw.f16.bin`),
    b(`${entryPath}/verification/latent_scaled.f16.bin`),
    b(`${entryPath}/verification/encoder_hidden_states.f16.bin`),
  ]);
  ctx.writeTensor(tensors.get("image", "sample"), nchwToNhwc16(scaledBuf, 1, 4, 64, 64));
  ctx.writeTensor(tensors.get("image", "latent_raw"), nchwToNhwc16(rawBuf, 1, 4, 64, 64));
  log(`image outputs bound: ${Object.entries(rig.graphs.image.outputs).map(([n, s]) => `${n}:${s.dataType}[${s.shape}]`).join(", ")}`);

  const hostBuf = new ArrayBuffer(512 * 512 * 4);
  const hostView = new Uint8Array(hostBuf);

  const dispatchText = () => ctx.dispatch(rig.graphs.text.graph, tensors.inputsFor("text"), tensors.outputsFor("text"));
  const dispatchImage = () => ctx.dispatch(rig.graphs.image.graph, tensors.inputsFor("image"), tensors.outputsFor("image"));
  const fenceText = () => ctx.readTensor(embedding); // completion fence only
  const readImage = () => ctx.readTensor(tensors.get("image", "out"), hostView);

  // ---- one correct run, checked -------------------------------------------
  t = T();
  dispatchText();
  const embBuf = await ctx.readTensor(embedding);
  R.timings.firstTextDispatchMs = +(T() - t).toFixed(1);
  {
    const got = new Float16Array(embBuf);
    const ref = new Float16Array(refEmbBuf);
    let maxAbs = 0, sum = 0, refMax = 0;
    for (let i = 0; i < got.length; i++) {
      const d = Math.abs(got[i] - ref[i]);
      if (d > maxAbs) maxAbs = d;
      sum += d;
      refMax = Math.max(refMax, Math.abs(ref[i]));
    }
    const spec = expected.expected.textGraph;
    R.checks.textGraph = {
      maxAbs, meanAbs: sum / got.length, refMaxAbs: refMax,
      tolerance: spec.tolerance, pass: maxAbs <= spec.tolerance,
    };
    if (!R.checks.textGraph.pass) R.errors.push(`text graph maxAbs ${maxAbs} exceeds ${spec.tolerance}`);
    log(`text graph vs reference embedding: maxAbs=${maxAbs.toExponential(3)} (tolerance ${spec.tolerance}) -> ${R.checks.textGraph.pass ? "PASS" : "FAIL"}`);
  }

  t = T();
  dispatchImage();
  await readImage();
  R.timings.firstImageDispatchMs = +(T() - t).toFixed(1);

  // sha256 of the RGBA int32 readback: the strongest available check.
  R.checks.outputSha256 = await sha256Hex(hostBuf);
  R.checks.expectedSha256 = expected.expected.imageGraph.sha256;
  R.checks.sha256Match = R.checks.outputSha256 === R.checks.expectedSha256;
  log(`out sha256: ${R.checks.outputSha256}${R.checks.expectedSha256 === "PENDING" ? "  (expected.json says PENDING)" : `  expected ${R.checks.expectedSha256} -> ${R.checks.sha256Match ? "MATCH" : "MISMATCH"}`}`);

  // Alpha must be opaque everywhere: the pack's -2^24 bias sets the high byte.
  {
    const px = new Uint8ClampedArray(hostBuf);
    let allOpaque = true;
    for (let i = 3; i < px.length; i += 4) if (px[i] !== 255) { allOpaque = false; break; }
    R.checks.alphaAllOpaque = allOpaque;
    if (!allOpaque) R.errors.push("alpha channel is not 255 everywhere");
  }

  // Draw it, and compare against both reference images.
  {
    const img = new ImageData(new Uint8ClampedArray(hostBuf.slice(0)), 512, 512);
    const cv = document.getElementById("out");
    cv.getContext("2d").putImageData(img, 0, 0);
    const [exp, ref] = await Promise.all([
      pngRGBA(`${entryPath}/verification/expected_image.png`),
      pngRGBA(`${entryPath}/verification/reference_image.png`),
    ]);
    R.checks.vsExpectedImage = psnrRGB(img.data, exp.data);
    R.checks.vsReferenceImage = psnrRGB(img.data, ref.data);
    R.checks.pngDataUrl = cv.toDataURL("image/png");
    log(`vs expected_image.png (the workbench's own pass-four run): PSNR ${R.checks.vsExpectedImage.psnrDb} dB, max byte diff ${R.checks.vsExpectedImage.maxByteDiff}`);
    log(`vs reference_image.png (native PyTorch MPS): PSNR ${R.checks.vsReferenceImage.psnrDb} dB, max byte diff ${R.checks.vsReferenceImage.maxByteDiff}`);
  }

  // ---- timing -------------------------------------------------------------
  const newPrompt = [], cached = [], imageOnly = [], textOnly = [];
  for (let i = 0; i < warmup; i++) { dispatchText(); await fenceText(); dispatchImage(); await readImage(); }

  for (let i = 0; i < runs; i++) {
    let t0 = T();
    dispatchText(); await fenceText();
    dispatchImage(); await readImage();
    newPrompt.push(T() - t0);

    t0 = T();
    dispatchImage(); await readImage();
    cached.push(T() - t0);

    t0 = T();
    dispatchText(); await fenceText();
    textOnly.push(T() - t0);

    t0 = T();
    dispatchImage(); await readImage();
    imageOnly.push(T() - t0);
  }
  R.timings.newPrompt = stats(newPrompt);
  R.timings.cachedPrompt = stats(cached);
  R.timings.textGraph = stats(textOnly);
  R.timings.imageGraph = stats(imageOnly);
  log(`new prompt ${R.timings.newPrompt.medianMs} ms | cached ${R.timings.cachedPrompt.medianMs} ms | text ${R.timings.textGraph.medianMs} | image ${R.timings.imageGraph.medianMs}`);

  // The output must not have drifted over 20 runs.
  R.checks.sha256AfterRuns = await sha256Hex(hostBuf);
  R.checks.stableAcrossRuns = R.checks.sha256AfterRuns === R.checks.outputSha256;
  if (!R.checks.stableAcrossRuns) R.errors.push("output changed between the first run and the last");

  R.log = lines;
  return R;
}


// ---------------------------------------------------------------------------
// kind "tokens": an encoder graph fed one input tensor per case, then the
// family's autoregressive decode loop; the bar is token-for-token identity
// with expected.json's ids on every case, and the timing is per case.
// ---------------------------------------------------------------------------
async function runTokens({ R, entry, entryPath, expected, runs, warmup, weightsBase, chunkMB }) {
  const T = () => performance.now();
  R.kind = "tokens";
  const familyBase = `/families/${entry.family}`;
  const family = await j(`${familyBase}/family.json`);
  const spec = family.contract?.chaining?.autoregressive;
  if (!spec) throw new Error(`${entry.family}/family.json has no contract.chaining.autoregressive block`);
  const manifest = await j(`${entryPath}/${entry.constants}`);
  const graphNames = Object.keys(entry.graphs).filter((k) => k !== "chain");
  const recipes = Object.fromEntries(await Promise.all(graphNames.map(async (g) => [g, await j(`${entryPath}/${entry.graphs[g].recipe}`)])));
  const encoderName = graphNames.find((g) => g !== spec.graph);
  if (!encoderName) throw new Error("tokens verification wants an encoder graph next to the decode graph");

  if (!navigator.ml) throw new Error("navigator.ml is missing: WebNN is not enabled in this browser");
  let t = T();
  const ctx = await navigator.ml.createContext({ deviceType: entry.target.backend.deviceType ?? "gpu" });
  R.timings.createContextMs = +(T() - t).toFixed(1);
  R.fingerprint = assertCoreMLFingerprint(ctx);
  log(`context: ${R.fingerprint.backend}, preferredInputLayout=${R.fingerprint.preferredInputLayout}`);
  log(`entry: ${entry.family}/${entry.id} (variant ${entry.variant}); graphs ${graphNames.join(", ")}; decode graph "${spec.graph}"`);
  ctx.lost?.then((i) => { R.contextLost = String(i?.message ?? i); log(`CONTEXT LOST: ${R.contextLost}`); }).catch(() => {});

  R.opCoverage = Object.fromEntries(graphNames.map((g) => [g, opCoverage(recipes[g])]));
  R.checks.opSupport = Object.fromEntries(graphNames.map((g) => [g, checkOpSupport(recipes[g], ctx)]));
  const missing = graphNames.flatMap((g) => R.checks.opSupport[g].missing);
  if (missing.length) throw new Error(`MLGraphBuilder is missing: ${[...new Set(missing)].join(", ")}`);
  log(graphNames.map((g) => `${g} ${Object.keys(R.opCoverage[g]).length} types / ${recipes[g].ops.length} ops`).join(", ") + " - all supported");

  const mkChunks = (bytes) => {
    if (!chunkMB) return null;
    const size = chunkMB * 1024 * 1024 + 1;
    const out = [];
    for (let o = 0; o < bytes; o += size) out.push({ byteOffset: o, byteLength: Math.min(size, bytes - o) });
    return out;
  };
  const sources = Object.fromEntries(Object.entries(manifest.constants).map(([key, c]) => [key, constantSource({ ...c, chunks: mkChunks(c.bytes) }, { baseUrl: weightsBase })]));
  t = T();
  const rig = await loadEntry(entry, sources, ctx, { baseUrl: entryPath, recipes });
  R.timings.loadEntryMs = +(T() - t).toFixed(1);
  R.graphStats = rig.stats;
  for (const g of graphNames) log(`${g} graph built: ${JSON.stringify(rig.graphs[g].stats)}`);

  // tokenizer, for the log and the decoded strings
  let tok = null;
  if (family.tokenizer?.module) {
    const mod = await import(`${familyBase}/${family.tokenizer.module}`);
    const Cls = mod.default ?? mod[family.tokenizer.class];
    tok = await Cls.load(`${familyBase}/tokenizer/`);
    R.checks.tokenizer = { stats: tok.stats?.() ?? null, parity: 0, cases: 0 };
  }

  const tensors = await createEntryTensors(ctx, rig);
  const encInputs = Object.keys(rig.graphs[encoderName].inputs);
  if (encInputs.length !== 1) throw new Error(`the encoder graph has ${encInputs.length} inputs; the tokens path feeds exactly one`);
  const imageT = tensors.get(encoderName, encInputs[0]);
  const dispatchEncoder = () => ctx.dispatch(rig.graphs[encoderName].graph, tensors.inputsFor(encoderName), tensors.outputsFor(encoderName));
  const tokensView = new Int32Array(rig.graphs[spec.graph].outputs[spec.tokensOutput].shape.reduce((a, b) => a * b, 1));
  const runCase = async (bytes) => {
    ctx.writeTensor(imageT, bytes);
    dispatchEncoder();
    return autoregressive(ctx, rig, tensors, spec, { tokensView });
  };

  // ---- every case once, checked ------------------------------------------
  const cases = [];
  for (const c of expected.images) cases.push({ ...c, bytes: new Uint8Array(await b(`${entryPath}/verification/${c.input}`)) });
  R.checks.tokens = { cases: cases.length, identical: 0, detail: [] };
  t = T();
  for (const c of cases) {
    const r = await runCase(c.bytes);
    const same = r.tokens.length === c.tokens.length && r.tokens.every((v, i) => v === c.tokens[i]);
    const first = same ? -1 : r.tokens.findIndex((v, i) => v !== c.tokens[i]);
    if (same) R.checks.tokens.identical++;
    const decoded = tok ? tok.decode(r.tokens) : null;
    const d = { id: c.id, nTokens: r.tokens.length, expected: c.tokens.length, identical: same, dispatches: r.dispatches, endedWithEos: r.endedWithEos,
      ...(same ? {} : { firstDiff: first, got: r.tokens.slice(Math.max(0, first - 2), first + 4), want: c.tokens.slice(Math.max(0, first - 2), first + 4) }),
      ...(tok && c.decodedClean !== undefined ? { decodedMatches: decoded === c.decodedClean } : {}) };
    R.checks.tokens.detail.push(d);
    if (tok && c.decodedClean !== undefined) { R.checks.tokenizer.cases++; if (d.decodedMatches) R.checks.tokenizer.parity++; }
    log(`${same ? "ok  " : "FAIL"} ${c.id.padEnd(20)} ${String(r.tokens.length).padStart(4)} tokens in ${r.dispatches} dispatches${same ? "" : ` (differs at ${first})`}`);
  }
  R.timings.firstPassMs = +(T() - t).toFixed(1);
  R.checks.tokens.pass = R.checks.tokens.identical === R.checks.tokens.cases;
  if (!R.checks.tokens.pass) R.errors.push(`${R.checks.tokens.cases - R.checks.tokens.identical} of ${R.checks.tokens.cases} cases differ from the reference tokens`);
  log(`tokens: ${R.checks.tokens.identical}/${R.checks.tokens.cases} identical -> ${R.checks.tokens.pass ? "PASS" : "FAIL"}`);

  // ---- timing: per case, median of `runs` -------------------------------
  for (let i = 0; i < warmup; i++) for (const c of cases) await runCase(c.bytes);
  const per = Object.fromEntries(cases.map((c) => [c.id, []]));
  let totalTokens = 0;
  for (let i = 0; i < runs; i++)
    for (const c of cases) {
      const t0 = T();
      const r = await runCase(c.bytes);
      per[c.id].push(T() - t0);
      if (i === 0) totalTokens += r.tokens.length;
    }
  const medians = Object.fromEntries(Object.entries(per).map(([id, a]) => [id, +median(a).toFixed(2)]));
  const meds = Object.values(medians);
  R.timings.perCase = medians;
  R.timings.newPrompt = { ...stats(meds), n: runs, note: "median over cases of each case's median per-image time (encoder + greedy decode + tokens readback)" };
  R.timings.tokensPerRun = totalTokens;
  R.timings.msPerToken = +(meds.reduce((a, x) => a + x, 0) / totalTokens).toFixed(4);
  // encoder alone, fenced on its first readable output
  const encOutName = Object.keys(rig.graphs[encoderName].outputs).find((n) => (entry.graphs[encoderName].outputs.find((o) => o.name === n)?.readable) !== false) ?? Object.keys(rig.graphs[encoderName].outputs)[0];
  const encT = [];
  for (let i = 0; i < runs; i++) { const t0 = T(); dispatchEncoder(); await ctx.readTensor(tensors.get(encoderName, encOutName)); encT.push(T() - t0); }
  R.timings.encoder = stats(encT);
  // stability: the first case again
  const again = await runCase(cases[0].bytes);
  R.checks.stableAcrossRuns = again.tokens.length === cases[0].tokens.length && again.tokens.every((v, i) => v === cases[0].tokens[i]);
  if (!R.checks.stableAcrossRuns) R.errors.push("tokens drifted across runs");
  log(`per image: median ${R.timings.newPrompt.medianMs} ms (min ${R.timings.newPrompt.minMs}, max ${R.timings.newPrompt.maxMs}); ${R.timings.msPerToken} ms/token; encoder ${R.timings.encoder.medianMs} ms`);
  return R;
}
