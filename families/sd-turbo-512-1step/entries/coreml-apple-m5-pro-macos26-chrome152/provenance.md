# Provenance: sd-turbo-512-1step, entry `coreml-apple-m5-pro-macos26-chrome152`

The ledger for this entry: what was folded into these two graphs, what was
tried and rejected with the number that killed it, and what the Core ML backend
turned out to be like. It is the part that does not survive in the op list.

Everything below was measured on the configuration in `entry.json`'s `target`:
WebNN / Core ML on an Apple M5 Pro, macOS 26.6.2, Chrome 152.0.7977.77. Another
configuration is another entry, and the numbers here say nothing about it.

SD-Turbo, 512x512, one Euler step, as **two** WebNN graphs for the Core ML
backend. Text encoder in one; UNet + the Euler step + the TAESD decoder + an
RGBA pack in the other. float16 throughout, no fp32 casts except one 1x1 conv
in the pack.

**67.85 ms** for a new prompt on an Apple M5 Pro, against 916 ms for ONNX
Runtime Web on WebGPU, 237 ms for the same demo on ORT's WebNN EP, and 94.5 ms
for native PyTorch MPS. Image PSNR 44.28 dB against the fp32 chain, where the
float16-on-MPS chain itself reaches only 45.36 dB.

| file | what |
|---|---|
| `entry.json` | the configuration this entry was built and tuned for, and its I/O |
| `recipe.image.json` | 1441 ops, 655 constants, 1650 MiB, NHWC |
| `recipe.text.json` | 603 ops, 281 constants, 649 MiB |
| `manifest.json` | the constants blobs, pinned by sha256, and the recipe hashes |
| `measurements.json` | every number on this page, machine-readable, one row per host |
| `verification/` | the reference case and this entry's bars |
| `../../tokenizer/` | CLIP BPE, shared by every entry of the family |

## What the recipes are

Two graphs, one `MLContext`, chained on device.

```
prompt --tokenize--> input_ids int32 [1,77]
                          |
                    [ text graph ]  603 ops
                          |
              encoder_hidden_states float16 [1,77,1024]   <-- one MLTensor,
                          |                                   never read to JS
latent_scaled  ----->     |
latent_raw     ----->[ image graph ]  1441 ops
                          |
                    out int32 [1,512,512]  <-- RGBA packed, one int32 per pixel
```

`latent_scaled` is `scale_model_input(latent_raw)`; `latent_raw` is bound
separately because the in-graph Euler step needs the unscaled latent. Both are
NHWC `[1,64,64,4]`; the reference bins on disk are NCHW, and the caller
converts. Full contract in `manifest.json`.

---

## Folds and rewrites that are in these graphs

Every delta below is an **interleaved A/B**: N complete configurations built in
one page session, all graphs resident, timed A,B,A,B in two blocks of 20 to 50.
A difference counts only if it exceeds 1 ms in *both* blocks, and every claimed
win needs a byte-identical position control in the last slot, because slot bias
alone reaches +-1.4 ms.

### The single dispatch

The Euler step is two ops inside the graph:
`latent = latent_raw + (-sigma_0) * unet_out`, with sigma_0 = 14.614646911621094 baked as
a float16 constant. The TAESD decoder is appended after it. So one dispatch
takes a latent to pixels, and the `[1,64,64,4]` intermediate never leaves the
device. This was worth landing at 67.5 ms in pass one.

### RGBA pack to one int32 per pixel: **-6.15/-6.45 ms**

uint8 cannot leave a Core ML graph: output dtypes are float32, float16 and
int32 only. So each pixel leaves as one int32,
`R + 256 * G + 65536 * B - 2^24`, whose low bits are `R | G<<8 | B<<16 | 0xFF000000`
so alpha is 255 for free.

TAESD's `conv_out` fold becomes `x255`, then `clamp(0,255)`, `roundEven`, cast
to float32, a 1x1 float32 conv with filter `[1,256,65536]` and bias `-2^24`, cast
to int32. The conv is float32 because 65536 exceeds float16's 65504.

On the JS side the readback is a `Uint8ClampedArray` view built once, so
`ImageData` is a permanent window onto it.

Where the win came from: readback 3.3 -> 0.3 ms, conversion 1.0 -> 0 ms, and the
image dispatch itself 65.45 -> 62.05 ms, because Core ML writing a
`[1,512,512]` int32 output costs about 2 ms less than a `[1,512,512,3]`
float16 one. Max RGBA byte difference against the float16 path: 1.

### Sub-pixel upsample, UNet only: **-1.1 / -1.6 / -1.7 / -2.4 ms**

The three `Upsample2D` sites are nearest-2x followed by a 3x3 conv. Rewritten
as four sub-kernels (1x1, 1x2, 2x1, 2x2) applied to the low-resolution input,
then depth-to-space spelled as a transpose-free reshape: concat columns, then
rows. Chrome 152's `MLGraphBuilder` has no `depthToSpace` or `pixelShuffle` and
rejects rank 6, so the reshape spelling is not a preference.

Exact in real arithmetic; the fp64 check's worst relative difference is
3.87e-16. Deletes 44% of the upsample conv's FLOPs and never materialises the
2Hx2W tensor. Costs +49 MiB of constants. UNet max abs *improved*, 3.87e-3 ->
3.41e-3.

Per-site, isolated: `up_blocks.1` (1280ch, 16 -> 32) 1.60 -> 1.10 ms;
`up_blocks.2` (640ch, 32 -> 64) 2.10 -> 1.60 ms; `up_blocks.0` (1280ch, 8 -> 16) was
+0.1, a loss too small to matter.

### LayerNorm to FF projection fold: **0.00 ms, kept for accuracy**

The 16 `norm3 -> ff.net.0.proj` instances have their LayerNorm gamma/beta
absorbed offline into the following linear weights. Zero ops and zero bytes
change, because the affine is free inside Core ML's fused `layer_norm`, and 32
constants disappear.

It is not a speed win. The raw delta was +0.15/+1.05 ms and the position
control measured +0.35/+1.4 ms, i.e. indistinguishable from slot bias.

It is kept because of what it does to the numbers:

| | before | after |
|---|---|---|
| UNet max abs | 3.41e-3 (0.50x the float16 floor) | **2.89e-3 (0.43x)** |
| image PSNR | 41.89 dB | **44.28 dB** |
| image max abs | 7.02e-1 | **3.20e-1** |

*Verification is a valid reason to keep a zero-cost rewrite.*

### NHWC: **the alternative was +3.4/+2.0 ms**

The whole graph is NHWC, though Core ML declares `preferredInputLayout` `nchw`.
The transformer's permute+reshape becomes a free reshape.

Global NCHW as a control deleted 43% of the transpose bytes (331.85 -> 188.66 MB)
and 108 ops and was **slower**. The loss is the pixel-shuffle spelling, not the
transposes. See the backend notes below.

### Carried forward from earlier passes

- **Packed QKV** with the attention scale folded into `to_q`: one matmul then
  `split(3, axis=1)`. Speed unchanged; the scale fold halved the attention error.
- **Timestep projection folded into `conv1.bias`.** This graph is
  single-timestep (t=999) by construction, so the time embedding is a
  compile-time constant and the whole subgraph is gone.
- **Linear as matmul, not 1x1 conv, in the text encoder:** 7.15 vs 11.8 ms.
- **Plain stride-2 conv downsamplers**, that is, the un-rewritten baseline. The
  alternatives were measured and rejected.

---

## Rewrites that were tried and rejected

| rewrite | the number | why |
|---|---|---|
| Hoist cross-attention K/V into the text graph | +1.75/+2.2 ms | the 32 hoisted matmuls were already fully hidden by Core ML's scheduling; moving them just adds text-graph cost |
| Attention without Core ML's softmax (exp, ones-column row sums) | +3.3/+4.1 and +2.55/+4.6 ms | Core ML's softmax is one fused pass; the rewrite is five. It is ~1 dB *more* accurate, so it survives as an accuracy option, not a speed one |
| Chunked softmax, Q in 2/4/8 chunks | -1.45/-0.25, -0.1/-0.3, +0.35/+1.0 | moves identical bytes; the softmax cost *is* the bytes, and the curve turns up |
| One transpose of the packed QKV | -1.0/+0.15 | 16 fewer transposes, and it is noise |
| TAESD decoder in NCHW | -0.75/+0.05 in chain | isolated it was 8.8 -> 6.1 ms, but its loss was the float16 output write, already removed by the RGBA pack |
| Sub-pixel upsample for TAESD's three stages | one stage -2, one +1 | no consistent win |
| Stride-2 downsampler as space-to-depth | new -1.6/+0.05, cached +1.1/+0.75; PSNR 44.28 -> 43.46 | the parity sub-kernels sum to exactly 9 taps: FLOP- **and** byte-neutral. It deletes nothing |
| Skip fusion `conv(concat(h,s)) -> conv_a(h) + conv_b(s)` | net +0.33 ms; all 12 concats are 0.25 ms of the plan | rejected on arithmetic before building; 4 of 12 instances are not even exact, as GroupNorm groups straddle the concat |
| Layout assignment as a min-cut over transpose bytes | not built | premise falsified: Core ML's transpose price is an analytic model, not a measurement |
| int8 weights, per-channel and blockwise-32 | +-0 ms; first dispatch 0.2 -> 35 s; max abs 1.2e-1 against a 1.36e-2 bar | Core ML expands int8 to float16 at load. It is a transfer format (1598 -> 832 MiB), not a compute format |
| int4 weights, blockwise-32 | +25 ms (44% slower); max abs 4.05e-1 | stays compressed and is dequantized every inference |
| `deviceType: "npu"` (CPUAndNeuralEngine) | 694 ms; PSNR 17.6 dB | the image graph is GPU-only and the ANE cannot hold it numerically. This also retires split-einsum: there is no ANE to target |
| Text encoder on the Neural Engine | 7.4 ms either way; embedding error 6.25e-2, 2.7x the float16 floor | 601/608 ops are ANE-eligible and it *does* compile for the ANE (build 4.8 -> 42.6 s), but there is no speed to trade for the accuracy loss |
| `MLOptimizationHints` (reshapeFrequency infrequent, fastPrediction) | +0.26/-1.09 then -1.12/+1.83, and the sign flips | no measurable effect; output bit-identical |
| WebNN to WebGPU interop | not run; rejected on source | macOS exports only float16 tensors, mutually exclusive with the int32 RGBA output worth ~6 ms |
| fp16 range scaling at seams | tightest softmax margin 1402x, norms 178x | no seam needs it |

---

## What this backend is actually like

These are the facts that shaped the recipes, and they generalise past this
model.

**Core ML fuses 45 of 1,536 ops, all `sigmoid + mul -> silu`.** No conv+add
fusion, no constant folding, no CSE, no dead-op elimination, no transpose or
reshape removal, no attention pattern match. *The graph you emit is the graph
that runs.* This is the reason almost every algebraic rewrite lost: decomposing
a fused kernel always costs, however exact the algebra.

**`softmax` is one fused kernel** (max, exp, sum, divide in a single pass) and
costs 3.1x its input matmul at every scale. That is the signature of pure memory
traffic over the `[5,4096,4096]` score tensor.

**`MLComputePlan`'s cost model is analytic, not measured.** Every transpose is
priced at 7.90 us/MB (NHWC) or 8.39 us/MB (NCHW), identical to three digits
regardless of permutation, shape or level. Deleting all 28 MB of Blink's
transposes from a ResNet block moves it <= 0.1 ms, with inconsistent sign. Use
the plan to find what is *big*; use an isolated-block harness to decide whether
a rewrite *wins*.

**`MLComputePlan` returns null cost for every row** of a model containing one
unestimable op, not just that op.

**`averagePool2d`** builds and runs standalone but fails with `Failed to get
compiled graph devices` whenever a second image graph is resident in the same
`MLContext`. Reproduced twice, only for this op.

**uint8 cannot leave a Core ML graph.** Output dtypes are float32, float16,
int32. This is what forces the RGBA int32 pack.

**Chromium has no compiled-graph cache.** `compileModelAtURL` runs on every
`build()`; the `.mlpackage` goes to a randomly named temp dir and both it and
the compiled `.mlmodelc` are deleted when the graph is destroyed. Not per
machine, not per page load, but per `build()` call. Reloading the page pays the
full ~25 s again.

**Core ML is gated on a non-incognito profile.** Off the record, WebNN silently
falls back to TFLite/XNNPACK on the CPU: no error, ~50x slower, and the graph
builds in milliseconds instead of seconds. `runtime/loader.js`'s `assertCoreMLFingerprint` exists for exactly this.

**Native Core ML on the identical graph runs 62.5 ms against Chrome's 62.**
The browser adds no overhead.

**Blink's own transform pipeline** inserts 318 layout transposes on the NHWC
path, folds 102 into constants and eliminates 85. Its bracket-elimination
whitelist excludes reshape and broadcasting binaries, so little else cancels.

### Where the time goes

Compiled plan, 1,587 ops after Blink's pipeline, 1,542 costed, **100% GPU**.

| by op | | by resolution | | by template | |
|---|---:|---|---:|---|---:|
| conv | 34.6% | 64x64 | 46% | attn1 | 26% |
| matmul | 23.8% | 32x32 | 23% | ResNet | 25% |
| softmax | 12.2% | 16x16 | 15% | FF | 17% |
| add | 8.3% | TAESD | 13% | TAESD | 13% |
| reshape | 4.6% | 8x8 | 3% | GroupNorm | 6% |
| transpose | 4.2% | | | Upsample2D | 4.4% |
| concat | 0.4% | | | attn2 | 4.4% |

678 GFLOPs at roughly 12 TFLOP/s, against a measured float16 vector peak near
13.

### The floor

**About 60 ms for the image graph on this backend.** conv and matmul are at
hardware efficiency; the five 64x64 softmaxes cannot be spelled cheaper at the
WebNN level, because Chromium exposes no fused attention op and so the
`[5,4096,4096]` scores must be materialised; and every cheap category has now
been *measured*, not modelled, at approximately zero.

Getting under 60 needs a new op in the WebNN surface or fewer FLOPs, not another
graph rewrite. Substituting a fused `scaled_dot_product_attention` for all 32
matmul-softmax-matmul triples was measured natively at **-15.5 ms (24% of the
image graph)** and came out **more accurate**, 45.53 vs 44.82 dB, because the
float16 score tensor is never materialised. That is 2.4 ms better than deleting
the triples outright. Chromium cannot emit it today, and the MIL op lacks the
`scale` argument this graph needs.

---

## The optional fast variant

Not an entry in this family, because no recipe for it exists yet. Recorded here
because the measurement is the point.

As of this entry there is **no IR dump of the ToDo graph**: the workbench
measured the variant end to end but never recorded its `MLGraphBuilder` call
sequence, so there is no recipe to store and no second entry in this family.
Recording one is the next thing to do; inventing one from these numbers is not.

**ToDo (Token Downsampling)**, `--todo 2 --todo-pool nearest --todo-levels 64`
in the workbench: the K/V of the five 64x64 `attn1` layers are nearest-subsampled
2x, pooled *before* `to_k`/`to_v` so the projections shrink too. The float16
score tensor goes 168 MB -> 42 MB.

**68.3 -> 54.6 ms**, i.e. -20%. Deltas -14.65/-12.80 ms new prompt and
-13.25/-11.85 cached.

Quality against the exact pipeline, mean over 8 prompts:

| | nearest | avg |
|---|---:|---:|
| PSNR | 24.32 dB | 20.87 dB |
| SSIM | 0.843 | 0.697 |
| CLIP image-image | 0.9928 | 0.9713 |
| CLIP image-text | 0.3179 | 0.3215 |

(CLIP image-text for the unmodified pipeline is 0.3205.)

Composition, colour and subject are preserved; fine detail is re-rolled: thin
rims, whiskers. `nearest` beats `avg` on every prompt and every metric. Adding a
32x32 level buys ~1.5 ms more and costs 5 dB.

This is not a lossless optimisation. It is a slightly different image for 20%
less time, and it is one octave more aggressive than anything the ToDo paper
measured: they use factor 2 at 1024x1024 / 16,384 tokens, this is factor 2 on a
4,096-token grid.

---

## Verifying

```bash
node scripts/verify.mjs --entry sd-turbo-512-1step/coreml-apple-m5-pro-macos26-chrome152 \
  --weights ../webnn-workbench/bench/webnn/ir
```

Checks the tokenizer against `verification/input_ids.json`, the text graph
against the reference embedding, the sha256 of the RGBA readback against
`verification/expected.json`, the PSNR against both reference images, and times
20 runs. See `verification/expected.json` for the bars and their rationale.

### What it gave, replaying these recipes

Three builds on the reference machine, two constant paths, all under an
unrelated compute job (load average 16 to 18), so
the absolutes are 0 to 5 ms pessimistic in the way the workbench documented.

| run | new prompt | cached | image graph | text graph |
|---|---:|---:|---:|---:|
| whole-file constants | 70.30 | 62.45 | 62.80 | 7.40 |
| chunked, 192 MiB ranges | 72.35 | 64.60 | 64.45 | 7.70 |
| whole-file, final | **68.20** | 61.15 | 61.25 | 7.25 |
| reference (quiet machine) | 67.85 | ~60 | ~62 | 7.15 |

All three produced the same 1,048,576-byte RGBA readback, byte for byte:
`5838ed1a...`. The replay itself costs 2 to 8 ms for 1441 ops, against the
3,569 ms the original emitter spent building the same graph; the 28 s is Core
ML compiling it, and Chromium caches none of that.

One thing to know before comparing against the workbench's own output PNG:
**this recipe declares two graph outputs**, `out` and the debug `latent`,
because it was recorded with `latentOutput` on. The 67.85 ms configuration
declared one. A different output list is a different Core ML graph, compiled
and scheduled differently, and float16 reductions reassociate, so the images
differ by a maximum of 2 in 8 bits (PSNR 60.9 dB), which is rounding, not a
different picture. The replay is nonetheless deterministic: the same sha256
comes back across separate builds and across the whole-file and chunked
constant paths.

---

## Sources

Everything here was measured in the `webnn-workbench` repo:
`docs/results.md`, `bench/results/pass2-summary.md`, `pass3-summary.md`,
`pass4-summary.md`, `pass3-subpixel.md`, `pass3-attention.md`,
`pass3-layout-targets.md`, and the compiled-plan dumps
`pass3-plan-image-*.md`. The recipes themselves were dumped by
`bench/webnn/ir-record.js`, a Proxy around `MLGraphBuilder` that records the
exact call sequence with the shapes the backend inferred, so a recipe cannot
silently disagree with the graph that was measured.
