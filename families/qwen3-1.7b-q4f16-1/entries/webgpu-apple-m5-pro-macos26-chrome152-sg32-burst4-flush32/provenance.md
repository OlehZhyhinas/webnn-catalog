# Qwen3-1.7B q4f16_1 WebGPU — catalog ledger

Exact-greedy Qwen3-1.7B tuned on Apple M5 Pro / Chrome 152.0.7977.77.
The authoritative experiment record is
`OlehZhyhinas/webnn-workbench@ca301a9c0154c9788ad5585a20048a5c2a850642`,
`docs/qwen17-m5-webgpu-report.md`.

## Shipped stack

- WebLLM 0.2.84 ABI, runtime source
  `OlehZhyhinas/web-llm-qwen@b9e2b01d3d3f4b5e1fad005da159e31430c65130`.
- MLC-LLM source
  `mlc-ai/mlc-llm@ed1c7f65f7fc4f08c53db1294c721f4c0ba49a35`.
- TVM/Relax source
  `mlc-ai/relax@c04f730addef05dda98f5ee23986de885ade9b9d`,
  plus the private compiler-worktree fix described in the workbench report.
- Upstream model repository
  `mlc-ai/Qwen3-1.7B-q4f16_1-MLC@80b3abcec6c3b3f5355dc0cc99cc4fb578f192bc`.
- Subgroup-32 model library, chunk-256 two-stage full-vocabulary argmax,
  GEMV `TR=32`, and K=4 GPU-resident exact-greedy decode.
- One WebGPU compute pass with a queue submit every 32 dispatches. Bind-group
  caching is disabled because it did not help this model.

Runtime JS and model-library WASM are public at immutable Hugging Face
revision `5adde901b11cd9faa70ca4c1eade2566dd4a00e8` and pinned again by byte
count and SHA-256 in `artifacts.json`. Weights and tokenizer stay in the
immutable upstream model revision.

## Methodology and measured result

Every browser run used a real Google Chrome persistent context, the dedicated
`bench/.chrome-profile-qwen17` profile, port 8907, and the shared machine GPU
lock. ORCA continued saturating the CPU; `WEBNN_MAX_LOAD=0` disabled load
gating rather than stopping or starving it.

The headline uses six rotating paired rounds. The model and GPU pipelines are
warm, but each measured observation calls `resetChat(false)` and uses a
different prompt, so this is not a prompt-cache benchmark. Generation uses
temperature zero and thinking disabled. Ratios are computed within each round
before taking the median.

- Published subgroup-32 median: **105.305 tokens/s**.
- Final stack median: **139.86 tokens/s** (**7.149 ms/token**).
- Median paired ratio: **1.371x**.
- Ratio IQR: 1.305–1.400; MAD 0.062; range 1.108–1.457.
- One-minute load-average range: 23.50–25.51.
- Six of six fresh outputs and four of four quality outputs were
  byte-identical to the published path.

The 6.5 ms/token / 155 tokens/s quiet figure is only a mechanical upper-bound
projection. Runtime overlap hides CPU encoding behind GPU work, so ORCA likely
makes this optimization more valuable than a quiet CPU would.

## Tuning record and negative results

The first pass verified compiler parity and then transferred the 0.6B stack:

- local kernels with GPU argmax disabled: 0.983x published, exact output;
- chunk-256 GPU argmax: 1.020x the fallback sampler path;
- `TR=64`: 0.990x `TR=32`; `TR=128`: 0.963x;
- K=4: 1.251x K=1; K=8 had no convincing gain and much higher variance.

The model-specific retuning then challenged every inherited choice:

- argmax chunks 64/128/256/512/1024: 256 remained the narrow winner;
- `TR=16` and 256: 0.882x and 0.875x the retained setting;
- GEMV unroll 16/32 compiled to the winner-identical WASM; 128/256 regressed;
- a fused embed+decode entrypoint: 1.008x with an IQR crossing parity and an
  additional shader;
- K=5: 1.016x K=4 over 12 fresh pairs but 0.985x on the quality corpus;
- K=6 and K=8: slower or substantially more variable.

GPU timestamps explained the result: the five decode projection GEMVs used
about 4.89 ms of a representative 6.44 ms GPU token, while full-vocabulary
argmax used about 0.11 ms. `TR=32` was faster for every named decode GEMV.

After the newer runtime-overlap work landed on workbench main, it was retuned
again for 1.7B:

- submit every 32 dispatches: 1.134x ordinary K=4;
- submit every 64: 1.099x;
- submit every 128: 1.107x;
- submit every 16: 1.008x versus 32, with an IQR crossing parity and twice
  the submit frequency;
- bind-group caching at cadence 64: 1.085x ordinary K=4 versus 1.105x
  without the cache.

The complete aggregate and per-round JSON, load averages, output hashes,
compiler issue, context scaling, artifact hashes, and WGSL inventory are in
the workbench commit above.

## Exactness boundary

The fast path is enabled only for temperature-zero requests without logprobs,
grammar or structured output, logit bias, custom logit processors, or
frequency/presence/repetition penalties. Every other request uses WebLLM's
ordinary one-step sampler. K=4 may delay streaming and cancellation by up to
four tokens; speculative KV entries after a stop are removed with scalar-typed
KV PopN calls.

Exactness here means deterministic greedy output equality on the recorded
ten-case corpus. It is not a claim of bit-identical intermediate logits or
stochastic-sampling parity.
