# Qwen3-0.6B q4f16_1 WebGPU, runtime overlap — catalog ledger

Second entry of this family for Apple M5 Pro / Chrome 152.0.7977.77. It keeps
the sibling entry `webgpu-apple-m5-pro-macos26-chrome152-sg32-burst4`
unchanged in every kernel, weight and burst setting and adds three WebGPU
runtime flags. The complete experiment record, including the ordered steps,
what was not done, and the knob table for an automated sweep, is in
`OlehZhyhinas/webnn-workbench@6ec63b8eeeb846380e302afd82397025cb2fdedd`,
`docs/results.md`, sections "Runtime overlap" and "Explicit record of this
pass".

## Shipped stack

- Model library: byte-identical to the sibling entry
  (`qwen3-0.6b-argmax-chunk256-sg32-tr32.wasm`, SHA-256 `89477de3…`),
  referenced at its existing published path.
- Runtime bundle: WebLLM 0.2.84 ABI, source
  `OlehZhyhinas/web-llm-qwen@b9e2b01d3d3f4b5e1fad005da159e31430c65130`.
  The bundle embeds npm `@mlc-ai/web-runtime` 0.26.0-dev0 patched in place by
  `scripts/patch-web-runtime-batch-pass.mjs`, `…-flush-every.mjs` and
  `…-bind-cache.mjs` in that repository. SHA-256 `d36ca3f7…067ccc`,
  6,593,519 bytes; the same bundle serves the Qwen3-1.7B `flush32` entry.
- Runtime flags, applied by `runtime/webllm-loader.js` before engine creation:
  `batchPass: true` (one `GPUComputePassEncoder` between flush points),
  `flushEvery: 64` (`queue.submit` every 64 dispatches), `bindGroupCache: true`
  (reuse `GPUBindGroup`s keyed by `GPUBuffer` identity). All three default off
  in the bundle, so the sibling entry's behaviour is unchanged by the shared
  runtime.

## Why the flags exist

Inside a K=4 burst the runtime encoded all ~440 dispatches of a decode step,
each in its own compute pass with a fresh bind group, and submitted nothing
until the sync. The GPU idled through the whole JS encode of every step, so
wall time was encode plus GPU time rather than the larger of the two. The
periodic submit lets the GPU execute the first dispatches while JS encodes the
rest; the single pass removes per-dispatch encoder boundaries; the cache
removes bind-group creation from the encode path.

## Measured result

Rotating interleaved paired rounds, browser and model reloaded per case per
round, six fresh prompts with KV reset before timing, output SHA-256 compared
to the baseline case in the same round. The baseline case in every run was
the sibling entry's exact stack run live, not its recorded numbers. Unrelated work kept
the CPU saturated (1-minute load 15–23) throughout.

- Headline (`qwen-fable-bindcache-orca.json`): **312.6 tokens/s** median,
  **1.243x** paired against the live sibling case (IQR 1.229–1.260,
  MAD 0.020, range 1.196–1.277, n=6). 6/6 byte-identical.
- Submit every 64 without the cache, three independent runs: 1.215x, 1.206x,
  1.224x. 18/18 byte-identical.
- Cadence sweep (paired medians): 8 → 1.178x, 16 → 1.206x, 32 → 1.208x,
  64 → 1.215x, 128 → 1.170x, 256 → 1.143x. Batch pass alone: 1.044x, 1.053x.
- K=8 with the same flags: 1.232x median but minimum 1.056x; not adopted.

Chained through the sibling entry's own paired K=4/K=1 ratio (1.692x, and
0.613x K=1/K=4 reproduced live in this pass), this entry is roughly 2.0x the
sibling's K=1 tuned library.

## Caveat

The overlap flags hide JS command encoding behind GPU execution, and a
saturated CPU inflates encode time. The paired 1.22–1.24x is therefore an
upper bound for a quiet machine; the floor is the ~1.05x of batch pass alone.
No quiet-machine absolute or projection is claimed. `measurements.json`
records only measured rows.

## Exactness boundary

Identical to the sibling entry: exact greedy only, K=4 fast path gated on
temperature 0 and no logprobs, grammar, logit bias, custom logit processor or
penalties, ordinary WebLLM sampler otherwise. The runtime flags change
command encoding and submission only; every kernel, weight and dispatch
sequence is the sibling's. `verification/corpus.json` carries the same ten
expected texts and hashes and is re-run by `npm run verify:webllm` against
this entry.

## Not done in this pass

No quiet-machine run, no repair or KV-reuse phases, no model-library
recompile, no prefill/TTFT work, no quality corpus beyond byte-identity with
the paired baseline. The flags are mutually exclusive with the runtime's
opt-in timestamp profiler, which falls back to one pass per dispatch.
