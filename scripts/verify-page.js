// The page half of scripts/verify.mjs. Everything that touches WebNN happens
// here; the Node driver only launches Chrome, serves files and prints a table.

import {
  loadRecipe,
  constantSource,
  assertCoreMLFingerprint,
  opCoverage,
  checkOpSupport,
} from "../runtime/loader.js";
import ClipTokenizer from "../models/sd-turbo-512-1step/tokenizer.js";

const MODEL = "../models/sd-turbo-512-1step";
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
  const { runs = 20, warmup = 5, weightsBase = "/weights", chunkMB = 0, prompt = null } = opts;
  const R = { opts: { runs, warmup, weightsBase, chunkMB }, errors: [], checks: {}, timings: {} };
  const T = () => performance.now();

  // ---- context + fingerprint ---------------------------------------------
  if (!navigator.ml) throw new Error("navigator.ml is missing: WebNN is not enabled in this browser");
  let t = T();
  const ctx = await navigator.ml.createContext({ deviceType: "gpu" });
  R.timings.createContextMs = +(T() - t).toFixed(1);
  R.fingerprint = assertCoreMLFingerprint(ctx);
  log(`context: ${R.fingerprint.backend}, preferredInputLayout=${R.fingerprint.preferredInputLayout}, rank max ${R.fingerprint.maxRank}`);
  ctx.lost?.then((i) => { R.contextLost = String(i?.message ?? i); log(`CONTEXT LOST: ${R.contextLost}`); }).catch(() => {});

  // ---- recipes ------------------------------------------------------------
  const [manifest, imageRecipe, textRecipe, expected, refIds] = await Promise.all([
    j(`${MODEL}/manifest.json`),
    j(`${MODEL}/recipe.image.json`),
    j(`${MODEL}/recipe.text.json`),
    j(`${MODEL}/verification/expected.json`),
    j(`${MODEL}/verification/input_ids.json`),
  ]);
  R.opCoverage = { image: opCoverage(imageRecipe), text: opCoverage(textRecipe) };
  const support = {
    image: checkOpSupport(imageRecipe, ctx),
    text: checkOpSupport(textRecipe, ctx),
  };
  R.checks.opSupport = support;
  if (!support.image.ok || !support.text.ok)
    throw new Error(`MLGraphBuilder is missing: ${[...support.image.missing, ...support.text.missing].join(", ")}`);
  log(`op coverage: image ${Object.keys(R.opCoverage.image).length} types / ${imageRecipe.ops.length} ops, text ${Object.keys(R.opCoverage.text).length} types / ${textRecipe.ops.length} ops - all supported`);

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
  const src = (key) => {
    const e = { ...manifest.constants[key], chunks: mkChunks(manifest.constants[key].bytes) };
    return constantSource(e, { baseUrl: weightsBase });
  };

  // ---- build both graphs --------------------------------------------------
  // Image first: it is the 1650 MiB one, and building it before the text
  // encoder's 649 MiB keeps peak resident bytes lower.
  t = T();
  const image = await loadRecipe(imageRecipe, src("image"), ctx, {
    onProgress: (p) => { if (p.phase === "constants") log(`  image constants ${(p.bytesRead / 1e6).toFixed(0)}/${(p.totalBytes / 1e6).toFixed(0)} MB, ${p.built}/${p.total}`); },
  });
  R.timings.imageLoadMs = +(T() - t).toFixed(1);
  R.imageStats = image.stats;
  log(`image graph built: ${JSON.stringify(image.stats)}`);

  t = T();
  const text = await loadRecipe(textRecipe, src("text"), ctx);
  R.timings.textLoadMs = +(T() - t).toFixed(1);
  R.textStats = text.stats;
  log(`text graph built: ${JSON.stringify(text.stats)}`);

  // ---- tokenizer ----------------------------------------------------------
  const tok = await ClipTokenizer.load(`${MODEL}/tokenizer/`);
  const usePrompt = prompt ?? refIds.prompt;
  const enc = tok.encode(usePrompt, 77);
  const idsMatch = usePrompt === refIds.prompt && refIds.ids.every((v, i) => enc.ids[i] === v);
  R.checks.tokenizer = { prompt: usePrompt, matchesReference: idsMatch, nTokens: enc.nTokens, stats: tok.stats() };
  if (usePrompt === refIds.prompt && !idsMatch) R.errors.push("tokenizer output does not match verification/input_ids.json");
  log(`tokenizer: ${enc.nTokens} tokens, matches reference ids: ${idsMatch}`);

  // ---- tensors ------------------------------------------------------------
  const S = 77, C = 1024;
  const inputIds = await ctx.createTensor({ dataType: "int32", shape: [1, S], writable: true });
  ctx.writeTensor(inputIds, enc.ids);

  // One tensor is the text graph's `out` AND the image graph's
  // `encoder_hidden_states`: the embedding never crosses into JS.
  const embedding = await ctx.createTensor({ dataType: "float16", shape: [1, S, C], readable: true });

  const [rawBuf, scaledBuf, refEmbBuf] = await Promise.all([
    b(`${MODEL}/verification/latent_raw.f16.bin`),
    b(`${MODEL}/verification/latent_scaled.f16.bin`),
    b(`${MODEL}/verification/encoder_hidden_states.f16.bin`),
  ]);
  const sample = await ctx.createTensor({ dataType: "float16", shape: [1, 64, 64, 4], writable: true });
  const latentRaw = await ctx.createTensor({ dataType: "float16", shape: [1, 64, 64, 4], writable: true });
  ctx.writeTensor(sample, nchwToNhwc16(scaledBuf, 1, 4, 64, 64));
  ctx.writeTensor(latentRaw, nchwToNhwc16(rawBuf, 1, 4, 64, 64));

  // Every declared graph output must be bound, `latent` included: the recipe
  // was recorded with the debug latent output on. Only `out` is readable.
  const outs = {};
  for (const [name, spec] of Object.entries(image.outputs))
    outs[name] = await ctx.createTensor({ dataType: spec.dataType, shape: spec.shape, readable: name === "out" });
  log(`image outputs bound: ${Object.entries(image.outputs).map(([n, s]) => `${n}:${s.dataType}[${s.shape}]`).join(", ")}`);

  const hostBuf = new ArrayBuffer(512 * 512 * 4);
  const hostView = new Uint8Array(hostBuf);

  const dispatchText = () => ctx.dispatch(text.graph, { input_ids: inputIds }, { out: embedding });
  const dispatchImage = () => ctx.dispatch(image.graph, { sample, encoder_hidden_states: embedding, latent_raw: latentRaw }, outs);
  const fenceText = () => ctx.readTensor(embedding); // completion fence only
  const readImage = () => ctx.readTensor(outs.out, hostView);

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
      pngRGBA(`${MODEL}/verification/expected_image.png`),
      pngRGBA(`${MODEL}/verification/reference_image.png`),
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
