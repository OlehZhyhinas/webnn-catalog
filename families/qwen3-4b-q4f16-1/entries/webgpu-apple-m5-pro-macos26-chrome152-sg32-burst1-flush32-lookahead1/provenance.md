# Qwen3-4B q4f16_1 WebGPU, K=1 with GPU lookahead — catalog ledger

Second entry of this family for Apple M5 Pro / Chrome 152.0.7977.77. It keeps
the sibling entry `webgpu-apple-m5-pro-macos26-chrome152-sg32-burst4-flush32`
unchanged in every kernel and weight (same model library, byte for byte) and
changes only the runtime: the same rebuilt WebLLM bundle already shipped by
the 1.7B family's `webgpu-apple-m5-pro-macos26-chrome152-sg32-burst1-flush32-lookahead1`
entry, whose greedy decode keeps one step queued on the GPU while the current
burst is read back, run here with a burst of one token instead of four. The
complete experiment record, including the ordered steps and the full
per-round data, is in
`OlehZhyhinas/webnn-workbench@c5dddad840d92b93318e2cc9a0df51b94afcd018`,
`docs/results.md`, section "Qwen3-4B second pass: K=1 with GPU lookahead
(2026-09-06)" and its "Explicit record of this pass". That commit sits on
branch `qwen4-fable` under workbench PR
[OlehZhyhinas/webnn-workbench#16](https://github.com/OlehZhyhinas/webnn-workbench/pull/16),
**which is not merged as of this writing**; the commit hash above is the
authoritative pin regardless of the PR's merge state.

## Shipped stack

- Model library: byte-identical to the sibling 4B entry
  (`qwen3-4b-argmax-chunk256-sg32-tr32.wasm`, SHA-256
  `61867ba3af26196f99e7091d044193b7069688e7e2733bd19f3a1a4a6f7a3b36`,
  5,844,976 bytes), referenced at the sibling's existing published URL.
  Subgroup-32, chunk-256 two-stage full-vocabulary argmax, GEMV `TR=32`.
- Runtime bundle: reused without duplication from the 1.7B family's lookahead
  catalog entry — `web-llm-0.2.84-qwen-m5-lookahead.js`, SHA-256
  `163177f1c2842ca60a52644109867c48e7ee6865530149a69ac3fd827d4cce35`,
  6,598,408 bytes. Source
  `OlehZhyhinas/web-llm-qwen@21698fd421d278ad0f7b3cc4e23abe3eaf4a042d`
  (`qwen-m5` branch). Nothing model-specific lives in this bundle; the same
  WebLLM 0.2.84 ABI artifact serves both families.
- Runtime flags, applied by `runtime/webllm-loader.js` before engine
  creation: `greedyBurst: 1`, `greedyArgmax: true`, `deferDecodeCleanup:
  false`, `argmaxChunk: 256`, `gemvThreadRows: 32`, `batchPass: true`,
  `flushEvery: 32`, `bindGroupCache: false`, `engineConfig: {}`,
  `lookahead: 1`.

## Why the change exists

Same mechanism as the 1.7B family's lookahead entry: inside the sibling's
K=4 burst the runtime issues four steps, then `device.sync()` drains the GPU
queue, four int32 readbacks land, tokens are detokenized and streamed, and
only then does JS encode the next step — the GPU idles through all of that
once per burst. With lookahead the runtime issues burst + 1 steps and awaits
only the burst's readbacks through the runtime's readback promise chain, so
the GPU always has a step queued; with that bubble gone the burst no longer
needs to be four, and K=1 wastes none of the stream while still streaming
every token. The one lookahead step past EOS is still computed; it lands
after the stream ends and a back-to-back request's prefill queues behind it.

A 4B decode step does roughly twice the GPU work of a 1.7B step, so the
fixed per-burst CPU-side cost that lookahead hides (readback wake-up,
detokenization, the first dispatches of the next step) is a smaller fraction
of each token here. That is the expected and observed reason this entry's
paired ratio is below the 1.7B sibling entry's 1.136x.

## Measured result

Rotating interleaved paired rounds, browser and model reloaded per case per
round, fresh prompt per round with KV reset before timing, output SHA-256
compared to the baseline case in the same round. ORCA kept the CPU saturated
throughout (`WEBNN_MAX_LOAD=0` disabled load gating rather than stopping or
starving it).

| Run | Rounds | Load avg (1-min) | What it isolates | Result |
| --- | --- | --- | --- | --- |
| B (headline) | 12, interleaved-ab | 18.4–37.5 | K=1 lookahead 1, shipped bundle, vs. the sibling entry's own stack run live in the same rotation on its own bundle `d36ca3f7` (K=4, batch pass, flush 32) | **78.50 tok/s** median (12.7397 ms/token) vs **74.17 tok/s** (13.4825 ms/token); **1.063x** paired (IQR 1.013–1.093, MAD 0.044, range 1.005–1.133); paired end-to-end (TTFT+decode) 1.041x; TTFT delta 0.0 ms; bundle swap alone (K=4/flush 32 on the lookahead bundle vs. the shipped bundle, same run) 1.008x (1.004–1.030) |
| A | 12, interleaved-ab | 16–22 | Lookahead variants, all cases on the lookahead bundle, baseline the sibling's K=4/flush 32 | K=1 lookahead 1: 1.058x (1.011–1.080); K=1 lookahead 2: 1.022x (1.010–1.069); K=2 lookahead 1: 1.009x (0.992–1.039); all 12/12 exact |
| C | 4, interleaved-ab | 25–30 | Quality corpus (math/prose/code/low-margin), decode-only, mixed prompt lengths | 16/16 quality outputs byte-identical to the sibling stack and equal to the sibling's four quality hashes; mixed-corpus decode 1.104x (1.099–1.112) |

Run B's 12/12 fresh outputs are byte-identical to the paired baseline case in
the same round, and identical to the six fresh-prompt hashes the sibling
entry lists against the published path, with equal completion-token counts
(47, 16, 42, 23, 51, 19).

## Caveat

Lookahead hides CPU-side gaps behind GPU execution, and a saturated CPU
widens those gaps, so the paired 1.063x is an upper bound for a quiet
machine. No quiet-machine absolute or projection is claimed;
`measurements.json` records only measured rows.

## Exactness boundary

Identical to the sibling entry: exact greedy only, the GPU-resident path is
gated on temperature 0 and no logprobs, grammar, logit bias, custom logit
processor or penalties, ordinary WebLLM sampler otherwise. The lookahead
changes when steps are issued and awaited, not what they compute; the one
speculative step past a stop is rolled back with `kv_state_popn`, and
`prefillStep`, `resetChat` and `dispose` drain anything an abort leaves
queued. `verification/corpus.json` carries the same ten expected texts and
hashes as the sibling and is re-run by `npm run verify:webllm` against this
entry.

## Not done in this pass

No quiet-machine run, no repair or KV-reuse phases, no prefill/TTFT
optimization beyond reporting the lookahead tail's TTFT cost, no quality
corpus beyond byte-identity with the paired baseline, no change to weights,
quantization, argmax chunk, `TR` or the attention kernel, and no new runtime
bundle — this entry reuses the 1.7B family's lookahead artifact unmodified.
