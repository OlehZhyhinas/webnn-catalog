# Texo (FormulaNet, 20M) as hand-built WebNN graphs

Results and working notes for [issue #1](https://github.com/OlehZhyhinas/webnn-workbench/issues/1).
Machine and protocol as in [scope.md](scope.md): Apple M5 Pro, macOS 26.6.2,
Chrome 152.0.7977.77 pinned, WebNN on the Core ML backend (`deviceType: "gpu"`,
`preferredInputLayout === "nchw"` asserted), a persistent Chrome profile.
Everything is under `bench/webnn/texo/`; results in `bench/results/texo-*.json`.

## Result

**26 ms median per image, 30 ms mean, over LatexGen's 18 benchmark images, with
tokens identical to the fp32 ONNX reference on all 18.** LatexGen today runs the
same model in 770 ms (ONNX Runtime WASM, fp32) or 780 ms (ORT WebGPU). The
issue's target was 30–60 ms and its acceptance bar a median under 100 ms.

| Runtime | Per image (median of 18 image medians) | Mean | Per token | Tokens identical |
|---|---|---|---|---|
| LatexGen, ONNX Runtime Web WASM, fp32 (as shipped) | ~700 (584–1014 recorded) | 770 | ~15 | reference |
| LatexGen, ONNX Runtime Web WebGPU (int4 tried) | 780 | | | |
| **WebNN Core ML, encoder + decoder unrolled 16 steps, cache 512** | **25.9 / 26.5** | **29.3 / 30.0** | **0.64** | **18/18** |
| same, cache 128 (fast variant, caps output at 128 tokens) | 24.1 / 23.9 | 27.0 / 26.9 | 0.58 | 18/18 |
| same, one decode step per dispatch (the issue's design as written) | 70.2 / 70.5 | 81.6 / 82.1 | 1.74 | 18/18 |

Two numbers per cell are the two interleaved blocks (A,B,A,B, n = 10 each, 3
warmups) of one page session, `bench/results/texo-final.json`, machine load
average 5. Per image the median ranges from 11.5 ms (`euler`, 11 tokens) to
63.7 ms (`gaussian-text`, 125 tokens). The encoder is 2.3 ms of that; the rest
is 0.64 ms per generated token. LatexGen's own per-image numbers come from
`bench/results-images-2026-09.json` in that repo, whose recorded strings the
graphs reproduce exactly after the standard `clean_up_tokenization`.

## The model, as LatexGen ships it (corrections to the issue text)

Read off `LatexGen/public/models/texo/` with `onnx` and `onnxruntime`, not
from the config (`bench/webnn/texo/export_texo.py`).

- **The encoder emits 144 tokens, not 4.** `encoder_model.onnx` declares
  `last_hidden_state` as `[batch, 4, 2048]`, but that is a stale trace shape:
  ORT warns about it and returns `[1, 144, 2048]` for a `[1,3,384,384]` input.
  HGNetv2 has total stride 32 (stem 4, three depthwise stride-2 downsamplers),
  so 384 → 12×12 = 144 positions × 2048 channels. Cross-attention runs over 144
  keys and the hoisted encoder-side K/V are `[1,16,144,24]` per layer per side.
- **Encoder** (`my_hgnetv2`): 80 Conv (BatchNorm already folded: every Conv
  carries a bias), 53 ReLU, one 2×2 stride-1 MaxPool in the stem, 10 Concat,
  2 residual Add (stage 3, blocks 1 and 2), and two stem `Pad`s whose amounts
  the ONNX graph computes dynamically but which are `(0,1,0,1)` at 384². Stages
  `[in, mid, out, blocks, layers, k, downsample, light]`:
  `[48,48,128,1,6,3,·,·]`, `[128,96,512,1,6,3,✓,·]`, `[512,192,1024,3,6,5,✓,✓]`,
  `[1024,384,2048,1,6,5,✓,✓]`. A light layer is a 1×1 conv (no activation) then
  a depthwise k×k conv then ReLU; the downsamplers have no activation; each
  block concatenates its input with its six layer outputs and squeezes through
  two 1×1 convs. 54.1 MB fp32 → 27 MB fp16.
- **Encoder → decoder projection.** The decoder ONNX has `enc_to_dec_proj`
  (2048 → 384, with bias) before each layer's cross-attention `k_proj`/`v_proj`;
  both are affine, so they compose into one 2048 → 384 projection per layer
  per side, exactly.
- **Decoder** (mBART, `decoder_model_merged.onnx`, one `If` on
  `use_cache_branch`): embed_tokens `[687,384]` × √384, learned positions
  `[1029,384]` at offset 2, `layernorm_embedding`, 2 pre-LN layers (self-attn
  16×24 with q scaled after projection, cross-attn, fc1 → exact-erf GELU → fc2
  at 1536), final `layer_norm`, lm head `[384,687]` without bias. Greedy, start
  0, eos 2, `max_new_tokens` 512. 25.8 MB fp32 → 12.9 MB fp16. The merged graph's
  cached branch emits empty `present.*.encoder.*`: encoder K/V are computed once.
- **Tokenizer**: `WordLevel` over `WhitespaceSplit`, 687 entries, template
  `<s> A </s>`, no normalizer; decoding is a table lookup joined with spaces,
  and LatexGen's recorded strings additionally have transformers'
  `clean_up_tokenization_spaces` applied (`" ,"` → `","`, `" !"` → `"!"`, …).
- **Preprocessing stays in the consumer.** `preprocess-dump.mjs` runs
  LatexGen's `texoPreprocess` verbatim in the benchmark Chrome on the 18 PNGs
  and dumps the `[1,1,384,384]` tensors; those bytes are the contract that both
  the fp32 reference and the graphs consume (`drawImage` resampling is
  Chrome's; a numpy re-implementation would differ in the last bit).

## Round-trip floor (measured first)

`bench/webnn/texo/floor.mjs`, `bench/results/texo-floor-{floor,cachevar}.json`.
Synthetic graphs, one page session, 100–200 timed steps each; `performance.now()`
is quantised to 0.1 ms here, so sub-0.1 ms numbers come from queued batches
divided by k.

| Probe | Per step | Notes |
|---|---|---|
| trivial: int32 `[1]` → add 1 → int32 `[1]`, write + dispatch + await read | 0.20 ms (p90 0.30) | build 130 ms |
| same, k dispatches queued then one read, k = 2…32 | 0.10–0.11 ms per dispatch | two disjoint tensor sets: 0.087 |
| readTensor alone, 4 bytes | 0.0 ms (below the timer) | |
| chain_k: k synthetic decode steps in one graph (gather → 384×1536 → gelu → 1536×384 → 384×687 → argMax → gather) | k=1 0.20, k=4 0.30, k=8 0.50, k=16 1.40 ms | 0.20 / 0.075 / 0.063 / 0.088 ms per token; deterministic |
| cache_io: four `[1,16,512,24]` fp16 caches in and out, ping-ponged, one-hot update in-graph | 1.6–2.0 ms | k=16 queued: 1.6 — **queuing does not amortise it** |
| two caches instead of four | 0.9 ms | |
| four caches, input only / output only | 0.77 / 1.1–1.2 ms | outputs cost more than inputs |
| four caches at S = 128 / 1024 | 0.72 / 2.8–3.1 ms | linear in bytes |

The dispatch itself is cheap (0.1 ms queued, 0.2 ms awaited) and a
data-dependent `argMax → gather` chain works on Core ML, so unrolling k decode
steps in one graph costs 0.06–0.09 ms per token of synthetic math. What is not
cheap is activation bytes crossing the graph boundary: about 0.4 ms per 384 KiB
tensor moved in and out (0.6–0.7 µs per KiB), paid on every dispatch whether or
not the readback is awaited. The static-cache design from the issue, one step
per dispatch with four full-length caches in and out, therefore has a floor of
1.6 ms per token at S = 512 before any arithmetic, which is what the k = 1 row
above measures (1.74 ms per token). Unrolling divides that by k.

## Design as built

Two graphs, one `MLContext`, chained through MLTensors (the encoder's four K/V
outputs are the decoder's inputs, never read to JS).

**Encoder graph** (`texo-encoder-webnn.js`, 155 ops, 31.8 MiB, NCHW, 2.3 ms):
`image` fp16 `[1,1,384,384]` in; the three identical input channels of stem1
summed into one; the stem's dynamic pads as constant pads; every conv with its
native bias; out come `enc_kT0/1` `[1,16,24,144]` and `enc_v0/1` `[1,16,144,24]`,
computed as four 1×1 convs on the final `[1,2048,12,12]` map with
`enc_to_dec_proj ∘ k/v_proj` folded into one weight each. The conv's
`[1,384,12,12]` output *is* `[1,16,24,144]`, K^T per head as the score matmul
wants it, for free; V takes one transpose. `input: "gray"` additionally folds
the `(x/255 − mean)/std` normalisation into stem1 and pads the raw image with
`mean` so the fold stays exact at the border; it verifies at 0.97× the fp16
floor and is offered as a second input contract.

**Decode graph** (`texo-decoder-webnn.js`, k = 16 steps per dispatch, 85 ops per
step, 10 MiB of constants shared by all steps): inputs `tok` int32 `[1]`, `step`
int32 `[1]`, `cache_kT0/1` `[1,16,24,S]` (K stored transposed), `cache_v0/1`
`[1,16,S,24]`, the four encoder K/V; outputs `tokens` int32 `[k]` and the four
updated caches, ping-ponged between two tensor sets. Per step: the position
embedding, the one-hot row and the causal-mask row come from ONE gather of a
`[S, 384+S+S]` table for all k positions of the dispatch, then a slice and a
split per step (Core ML lowers every gather with index sanitisation, five ops
each); token embedding gather from a table pre-scaled by √384; packed QKV matmul
with 1/√24 folded into the Q third; with one token per step every per-head
reshape is a view, so the step has no transposes; cache update
`cache + kT_new ⊗ onehot` (mul + add, exact, verified row by row); scores over
the full cache plus the gathered mask row; two-layer mBART body; final LN; lm
head; `argMax` → int32, which feeds the next step's gather. Steps past `<eos>`
compute garbage that JS discards.

Verification, image `pythagorean` unless stated, references from
`export_texo.py` (fp32 ORT / torch CPU; floor = the same graph in fp16 on MPS):

| Check | Ours | fp16 floor | Ratio |
|---|---|---|---|
| encoder `last_hidden_state` max abs / mean abs | 9.77e-2 / 6.2e-4 | 9.03e-2 / 5.1e-4 | 1.08× / 1.22× |
| encoder, `input: "gray"` fold | 8.74e-2 / 5.5e-4 | | 0.97× / 1.09× |
| encoder taps stem1 … stage4, max abs | 2.9e-4 … 9.8e-2 (rel L2 ≤ 8.4e-3) | | |
| hoisted `enc_kT`/`enc_v` vs ORT `present.*.encoder.*`, rel L2 | 4.1e-3 … 6.1e-3 | | |
| decoder logits at steps 0, 1, 9, 17, max abs (rel L2) | 5.1e-2 … 6.9e-2 (3.1e-3 … 4.5e-3) | | |
| caches after those steps, max abs | ≤ 2.0e-2 | | |
| greedy tokens, 18 images, every variant below | 18/18 identical to ORT fp32 | torch fp16 chain also 18/18 | |

The smallest top-1/top-2 logit margin in the reference sequences is 0.090
(`newton-text`); our logits differ from fp32 by up to 0.07, so parity is not
guaranteed by construction, it is measured, and it held on every variant and
session. Any change to the decoder's numerics has to re-run the 18-image check.

## What was measured, kept, and rejected

All deltas are interleaved A/B in one page session (two blocks; a delta counts
only if it exceeds 1 ms in both), mean over the 18 image medians unless noted.

| Lever | Delta per image | Verdict |
|---|---|---|
| unroll k = 4 / 8 / 16 vs k = 1 (`texo-ksweep`, busy machine) | −59/−61, −70/−73, −76/−78 ms | **kept, k = 16** |
| cache length 256 vs 512 at k = 16 | −0.6/−1.5 | noise |
| cache length 128 vs 512 at k = 16 | −2.4/−3.1 | real, offered as the fast variant; caps output at 128 tokens (LatexGen's contract is 512) |
| position/one-hot/mask rows from one table gather per dispatch vs three gathers per step | −1.2/−1.3 | kept; Core ML ops per step 107 → 91 |
| `scatterND` cache update instead of one-hot outer product (k = 8, 6 images) | +3.6/+3.4 | rejected |
| linears as 1×1 conv with native bias (one op instead of matmul + add) | +3.5/+3.6 | rejected: the conv costs more than the add it removes |
| decoder on a `cpu` context (MLComputeUnitsCPUOnly), K/V crossed through JS | per token 1.10 vs 1.18–1.24 ms, but cache zeroing 8.3 ms per image | rejected |
| weight constants shared across unrolled steps | dispatch unchanged; constants 129 → 10 MiB, build 5.7 → ~1.4 s | kept |
| `input: "gray"` normalisation fold | encoder 2.2 vs 2.3 ms, noise | offered, exact |

**Where the time goes now** (k = 16, S = 512): encoder 2.3 ms, then 0.64 ms per
token, of which 0.1 is the caches crossing the graph boundary (1.6 ms per
dispatch of 16) and ~0.55 is the step's 85 WebNN ops (91 Core ML ops). The
compiled plan (`bench/results/texo-plan-texo-decoder-k8.md`) puts every op on
the GPU and attributes 74% of its *estimated* cost to the eight full-cache
`mul`/`add` ops per step, but the cache-length sweep says otherwise: shrinking
those tensors 4× saved 10%. The step is bound by per-op launch overhead, about
6–7 µs per op regardless of size, so the remaining lever is op count per step,
not bytes or FLOPs (0.09 GFLOP per 8 steps).

Not done: recompute-without-cache (moves only 2 KB of ids per dispatch but
repeats the prefix's arithmetic every step, and per-op cost, not bytes, is the
bound); NPU (`WebNNCoreMLExplicitGPUOrNPU`) for the decoder; fewer ops per step
by merging the two layers' independent prep work further.

## Reproduce

```bash
node bench/webnn/texo/preprocess-dump.mjs          # 18 inputs from LatexGen's PNGs, via Chrome
.venv/bin/python bench/webnn/texo/export_texo.py   # weights + references, ORT parity, fp16 floor
node bench/webnn/texo/floor.mjs                    # the round-trip floor probes
node bench/webnn/texo/run-texo.mjs --k 16 --runs 10 --blocks 2   # verify + time, 18 images
node bench/webnn/texo/run-texo.mjs --dump-ir encoder=texo-encoder --dump-ir decoder=texo-decoder-k16 --k 16
```

Catalog entry: `webnn-catalog/families/texo-384/` (see its `provenance.md`).
