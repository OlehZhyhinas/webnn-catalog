# Qwen3-4B q4f16_1 WebGPU — catalog ledger

Exact-greedy Qwen3-4B tuned on Apple M5 Pro / Chrome 152.0.7977.77.
The authoritative experiment record is
`OlehZhyhinas/webnn-workbench@2744881fe3af9b2003df1fa8809779c0784c6399`,
`docs/qwen4-m5-webgpu-report.md`.

## Shipped stack

- WebLLM 0.2.84 ABI, runtime source
  `OlehZhyhinas/web-llm-qwen@b9e2b01d3d3f4b5e1fad005da159e31430c65130`.
- MLC-LLM source
  `mlc-ai/mlc-llm@ed1c7f65f7fc4f08c53db1294c721f4c0ba49a35`.
- TVM/Relax source
  `mlc-ai/relax@c04f730addef05dda98f5ee23986de885ade9b9d`,
  plus the isolated compiler-worktree fixes described in the workbench report.
- Upstream model repository
  `mlc-ai/Qwen3-4B-q4f16_1-MLC@a5c9fab855e3ccbdfed2e7e69683d75f30332161`.
- Subgroup-32 model library, chunk-256 two-stage full-vocabulary argmax,
  GEMV `TR=32`, and K=4 GPU-resident exact-greedy decode.
- One WebGPU compute pass with a queue submit every 32 dispatches.
  Bind-group caching is disabled.

The selected model-library WASM is public at immutable Hugging Face dataset
revision `170ebdf3b1707f939833511cf1d295305227c921` and pinned again by byte
count and SHA-256 in `artifacts.json`. The unchanged runtime JS is reused from
the existing 1.7B artifact at that same immutable dataset revision rather than
duplicated. Weights and tokenizer stay in the immutable upstream model
revision.

## Methodology and measured result

Every browser run used a separate real Google Chrome persistent context, the
dedicated `bench/.chrome-profile-qwen4` profile, port 8908, and the shared
machine GPU lock. ORCA continued saturating the CPU; `WEBNN_MAX_LOAD=0`
disabled load gating rather than stopping or starving it.

The headline uses six rotating paired rounds. The model and GPU pipelines are
warm, but each measured observation calls `resetChat(false)` and uses a
different prompt, so this is not a prompt-cache benchmark. Generation uses
temperature zero and thinking disabled. Ratios are computed within each round
before taking the median.

- Published WebLLM 0.2.84 subgroup-32 median: **57.74 tokens/s**.
- Final stack median: **73.85 tokens/s** (**13.541 ms/token**).
- Median paired ratio: **1.285x**.
- Ratio IQR: 1.255–1.296; MAD 0.021; range 1.236–1.313.
- One-minute load-average range: 17.98–19.64.
- Six of six fresh outputs, including completion-token counts, and four of
  four quality outputs were byte-identical to the published path.

There is no historical quiet 4B number. No quiet measurement or projection is
claimed.

## Model-specific tuning and negative results

The 1.7B stack was used only as a starting point; each retained choice was
remeasured on 4B:

- local compiler parity with GPU argmax disabled: 1.014x published, exact;
- chunk-256 GPU argmax: 1.010x fallback;
- chunk 64 initially appeared 1.019x faster, then reversed to 0.967x over
  eight direct confirmation rounds;
- chunks 128/512/1024 did not beat 256;
- `TR=16/64/128`: 0.827x/0.994x/0.966x `TR=32`;
- `TR=256` was not run because 16 and 128 clearly regressed;
- K=4: 1.263x K=1 in the expanded sweep;
- K=5 and K=8 did not beat K=4;
- K=6 was an unstable 1.036x in a 12-round fresh confirmation and reversed
  to 0.981x on the quality corpus;
- flush 32: 1.091x ordinary K=4;
- flush 16: 0.971x flush 32;
- flush 64: inconclusive 1.011x with a confirmation IQR crossing parity;
- flush 128 did not help;
- bind-group caching: inconclusive 1.016x with its IQR crossing parity.

The GPU profile was dominated by quantized GEMV. It did not show a compelling
host/embed seam, so a fused entrypoint was not added. q4f16 remained clearly
suitable, so the conditional precision-frontier experiment was not run.

The complete aggregate and per-round JSON, load averages, output hashes,
compiler issue, context scaling, artifact hashes, and 4B-specific WGSL
inventory are in the workbench commit above.

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
