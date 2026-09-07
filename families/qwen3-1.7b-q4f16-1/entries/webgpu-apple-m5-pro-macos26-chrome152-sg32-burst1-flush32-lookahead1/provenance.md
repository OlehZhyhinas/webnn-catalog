# Qwen3-1.7B q4f16_1 WebGPU, K=1 with GPU lookahead — catalog ledger

Second entry of this family for Apple M5 Pro / Chrome 152.0.7977.77. It keeps
the sibling entry `webgpu-apple-m5-pro-macos26-chrome152-sg32-burst4-flush32`
unchanged in every kernel and weight (same model library, byte for byte) and
changes the runtime: a rebuilt WebLLM bundle whose greedy decode keeps one
step queued on the GPU while the current burst is read back, and a burst of
one token instead of four. The complete experiment record, including the
ordered steps, the invalidated run, what was not done, and the knob table
for an automated sweep, is in
`OlehZhyhinas/webnn-workbench@1ff3cdaaab10a9ce740db208974974552f2ca719`, `docs/results.md`, section
"Qwen3-1.7B second pass: GPU lookahead across the burst readback" and its
"Explicit record of this pass".

## Shipped stack

- Model library: byte-identical to the sibling entry
  (`qwen3-1.7b-argmax-chunk256-sg32-tr32.wasm`, SHA-256 `305bacef…74cc`),
  referenced at its existing published path. Subgroup-32, chunk-256
  two-stage full-vocabulary argmax, GEMV `TR=32`.
- Runtime bundle: WebLLM 0.2.84 ABI, source
  `OlehZhyhinas/web-llm-qwen@21698fd421d278ad0f7b3cc4e23abe3eaf4a042d`
  (`qwen-m5`). It embeds npm `@mlc-ai/web-runtime` 0.26.0-dev0 patched in
  place by `scripts/patch-web-runtime-{profile,batch-pass,flush-every,bind-cache,readback-tail}.mjs`
  (`patch-web-runtime-all.mjs` applies and asserts all five). SHA-256
  `163177f1…4cce35`, 6,598,408 bytes. The headline was measured on the
  previous build of the same branch (`4aa42f1`, bundle `2bd7fec7…0e6bb`);
  `21698fd` changes only `unload()` teardown order (sync before dispose, so
  pending lookahead readbacks settle before the device is destroyed, which
  the catalog verifier exposed). A six-round paired sanity run on the shipped
  bundle measured 1.137x (IQR 1.091–1.184, 6/6 byte-identical).
- Runtime flags, applied by `runtime/webllm-loader.js` before engine
  creation: `greedyBurst: 1`, `lookahead: 1`, `batchPass: true`,
  `flushEvery: 32`, `bindGroupCache: false`. `lookahead` is new in this
  entry; a loader that predates the field ignores it and runs a plain K=1
  GPU-argmax decode on the same bundle.

## Why the change exists

Inside the sibling's K=4 burst the runtime issued four steps, then
`device.sync()` drained the GPU queue, four int32 readbacks landed, tokens
were detokenized and streamed, and only then did JS encode the next step.
The GPU idled through all of that once per burst; timestamp profiles put the
GPU work at ~6.4 ms of the 7.15 ms token. With lookahead the runtime issues
burst + 1 steps and awaits only the burst's readbacks through the runtime's
readback promise chain, so the GPU always has a step queued. With that
bubble gone the burst no longer needs to be four: a burst of K computes up
to K−1 steps past EOS before the readback reveals it, and K=1 wastes none
inside the stream while streaming every token. The one lookahead step past
EOS is still computed; it lands after the stream ends and a back-to-back
request's prefill queues behind it (~4 ms of TTFT in the measured runs).

## Measured result

Rotating interleaved paired rounds, browser and model reloaded per case per
round, fresh prompt per round with KV reset before timing, output SHA-256
compared to the baseline case in the same round. The baseline case in every
run was the sibling entry's exact stack run live (same bundle, K=4, batch
pass, flush 32), not its recorded numbers. Unrelated work kept the CPU saturated
(1-minute load 17–24 in the headline run).

- Headline (`qwen-qwen17-fable-kxl-orca.json`, 12 rounds): **158.66
  tokens/s** median (6.30 ms/token), **1.136x** paired against the live
  sibling case (IQR 1.065–1.189, MAD 0.067, range 0.999–1.434, n=12).
  12/12 byte-identical. Sibling case in the same run: 140.24 tokens/s
  (7.13 ms/token). Paired end-to-end time (TTFT + decode): 1.068x; TTFT
  +3.9 ms median.
- Same run: K=2 lookahead 1 1.116x, K=1 lookahead 2 1.174x (IQR 1.072–1.226,
  TTFT +9.6 ms, end-to-end 1.036x), K=2 lookahead 2 1.120x, K=3 lookahead 1
  1.081x. Lookahead 2 is the alternative for long-output workloads where
  the extra speculative step amortizes.
- Earlier runs of the pass (lookahead at K=4): 1.030–1.032x over 6 and 12
  rounds; K=4 lookahead 1 without batch pass and flush 1.007x, so the
  periodic submit is still needed; bind cache 1.001–1.025x (noise band, not
  adopted); flush 64 a wash; flush 16 0.981x. Quality corpus, 4 rounds,
  K=4 lookahead 1: 1.044x, 16/16 byte-identical.
- GEMV schedule probe under this roll (`TS`, `VEC_C`, shared activation):
  0.981–1.004x, all byte-identical; the model library is unchanged.

A later paired run (2026-09-07) measured this stack against the published WebLLM 0.2.84 runtime and subgroup-32 model library directly, six rotating rounds with a fresh prompt each, one-step sampler on the published side. The tuned stack decoded at 169.605 tokens/s against 105.735 for the published path, a paired median of 1.587x (quartiles 1.575 to 1.603), with byte-identical output in 6 of 6 rounds. The CPU carried an unrelated load of 13.5 to 17.0 throughout; the paired design is what makes the ratio usable under it.

## Caveat

Lookahead hides CPU-side gaps (readback wake-up, detokenization, the first
dispatches of the next step) behind GPU execution, and a saturated CPU
widens those gaps, so the paired 1.136x is an upper bound for a quiet
machine. The burst-tail component (~4–9% for the 9–32 token outputs of this
corpus) is load-independent but shrinks with output length. No quiet-machine
absolute or projection is claimed; `measurements.json` records only measured
rows.

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
optimization (TTFT is reported only as the cost of the lookahead tail), no
quality corpus beyond byte-identity with the paired baseline, no change to
weights, quantization, argmax chunk, `TR` or the attention kernel. One run
of the pass is kept on record as invalid because a pre-commit hook in the
runtime repository reinstalled `node_modules` and rebuilt the bundle under
it; the harness now records the served bundle's patch markers and refuses
to summarise a run whose observations span more than one bundle hash.
