# Qwen3.5-9B q4f16_1 WebGPU — catalog ledger

Exact-greedy Qwen3.5-9B tuned on Apple M5 Pro / Chrome 152.0.7977.77.
The authoritative experiment record is
`OlehZhyhinas/webnn-workbench@7bc7d4778f4bf9f6b1cba51b5b6c19e1e6fdf5c1`,
`docs/qwen35-9b-m5-webgpu-report.md`.

## Shipped stack

- WebLLM 0.2.84 ABI, runtime source
  `OlehZhyhinas/web-llm-qwen@21698fd421d278ad0f7b3cc4e23abe3eaf4a042d`.
- MLC-LLM source
  `mlc-ai/mlc-llm@ed1c7f65f7fc4f08c53db1294c721f4c0ba49a35`.
- TVM/Relax source
  `mlc-ai/relax@c04f730addef05dda98f5ee23986de885ade9b9d`,
  plus the isolated compiler-worktree fixes described in the workbench report.
- Upstream model repository
  `mlc-ai/Qwen3.5-9B-q4f16_1-MLC@c7c5d3f5a81e37b8facbb72970940a1b131314a8`.
- Subgroup-32 model library, chunk-256 two-stage full-vocabulary GPU argmax
  (the compiler's own default for this model), and K=4 GPU-resident
  exact-greedy decode.
- One WebGPU compute pass with a queue submit every 64 dispatches.
  Bind-group caching is disabled. Lookahead is not part of this headline
  (measured and rejected, see "Lookahead second pass" below).

The selected model-library WASM is uploaded to this entry's own path in the
`ozhyhinas/webnn-catalog` Hugging Face dataset and pinned by byte count and
SHA-256 in `artifacts.json`. The unchanged lookahead-capable runtime JS is
reused from the existing 1.7B lookahead-pass artifact at revision
`2efcfd5804a39bc2c1af40389e16d8a0a4e73d4a` rather than duplicated; this
entry runs it with `lookahead: 0`. Weights and tokenizer stay in the
immutable upstream model revision. Catalog runs used
`context_window_size` 4096.

## Architecture-specific finding: RNNState burst rollback

Qwen3.5 is a GatedDeltaNet hybrid (75% recurrent linear-attention layers
holding an `RNNState`, 25% GQA layers on the paged KV cache). The published
model record sets `max_history_size: 1`, which caps the RNN state's
rollback budget (`available_history_num`) at zero. A greedy burst that
overshoots a stop by even one token then crashes calling
`vm.builtin.kv_state_popn` on the RNN state
(`RNNStateImpObj::PopN: n <= available_history_num`). This is a runtime
`ChatConfig` field, not a property of the compiled WASM, so any consumer of
this entry's model library **must** raise
`model.config.overrides.max_history_size` to at least `burst + 1` (5, for
this entry's K=4) before creating the engine, or bursts will crash on
essentially every prompt. The loader (`runtime/webllm-loader.js`) reads
that override path directly when it builds the WebLLM model record; the
mirrored `runtime.config.maxHistorySize` field on the entry is
informational only and has no effect by itself. With the override in
place, a continued-generation (no-reset) check confirmed the RNN state and
paged KV cache both roll back correctly and produce byte-identical output
across a second turn.

## Methodology and measured result

Every browser run used a separate real Google Chrome persistent context, the
dedicated `bench/.chrome-profile-qwen35-9b` profile, port 8913, and the shared
machine GPU lock. Three sibling model rolls (Qwen3.5-0.8B, MiniCPM5-2B,
Qwen3.5-4B) shared the same machine and GPU lock throughout this run;
`WEBNN_MAX_LOAD=0` disabled load gating rather than stopping or starving them.

The headline uses eight rotating paired rounds. The model and GPU pipelines
are warm, but each measured observation calls `resetChat(false)` and uses a
different prompt, so this is not a prompt-cache benchmark. Generation uses
temperature zero and thinking disabled. Ratios are computed within each
round before taking the median.

- Published WebLLM 0.2.84 subgroup-32 median: **34.83 tokens/s**.
- Final stack median: **37.45 tokens/s** (**26.70 ms/token**).
- Median paired ratio: **1.070x**.
- Ratio IQR: 1.032-1.163.
- One-minute load-average range: approximately 15-20 (heavier than the
  loads recorded for the smaller Qwen3 sizes when only that one
  model-roll benchmark held the GPU lock, because here three other
  model-roll benchmarks shared it).
- Eight of eight fresh outputs, including completion-token counts, were
  byte-identical to the published path.
- Four of four quality outputs were byte-identical to local K=1
  full-vocabulary argmax. The published temperature-zero sampler alternated
  on the prose item, so that item is not claimed as a published-path match.

There is no historical quiet 9B number. No quiet measurement or projection
is claimed.

## Model-specific tuning and negative results

The 8B stack was used only as a starting point; each retained choice was
remeasured on 9B, and one new architecture-specific issue (above) had to be
fixed before any burst measurement was possible:

- local compiler parity with GPU argmax disabled: 1.010x published `sg32`,
  within the ~2% tolerance;
- chunk-256 GPU argmax (the compiler's own default for this model): 0.999x
  fallback, kept because it enabled the burst win;
- chunk 128 and 512 both initially appeared to lead (1.024x/1.019x), then
  both reversed to parity-crossing IQRs over eight confirmation rounds;
- chunks 64/1024 did not beat 256;
- `TVM_WEBGPU_GEMV_TR` produced **byte-identical WASM** for every value
  tried (16/32/64/128/default) — a verified no-op for this model's
  compiled kernels, not a tuning choice. The apparent 1.01-1.02x
  sweep/confirmation deltas were noise on an identical binary, retracted
  rather than reported as a win;
- K=4: 1.040-1.081x K=1 across three independent measurements (n=4, n=8,
  n=8), retained as a repeatable gain despite the 25th percentile touching
  parity under heavy concurrent load in each measurement;
- K=5 and K=8 both regressed (1.008-1.012x and 0.881-0.939x K=1); K=8's
  regression is sharper than at 8B, plausibly because 75% of layers are
  recurrent and pay a larger per-speculative-step RNN-state cost;
- flush 64: 1.017x ordinary K=4, IQR (1.012-1.035) fully clear of parity;
- flush 16: 0.998x flush 32;
- flush 128: apparent 1.006x, reversed to a parity-crossing IQR on
  confirmation;
- bind-group caching: apparent 1.029x, reversed to a parity-crossing IQR on
  confirmation.

A continued-generation (no-reset) exactness check — required by this
model's hybrid architecture — matched K=1 and the final K=4/flush64 roll
byte-for-byte on every round. The GPU profile showed quantized GEMV/GEMM
dominant with the GatedDeltaNet recurrent kernel (`gdn_func_kernel`) a
visible secondary contributor, and did not show a compelling host/embed
seam.

## Lookahead second pass

K=1 with GPU lookahead=1, paired live against the shipped K=4/flush64 roll
on the same bundle and library: 1.094x on 4 rounds, reversed to **1.028x**
on an 8-round confirmation (IQR 1.009-1.107, clear of parity). Below the
1.05x adoption bar used across this model family. **Not adopted** — the
same decision as 8B (there: 1.043x, also rejected). This entry's runtime
config keeps `lookahead: 0`.

The complete aggregate and per-round JSON, load averages, output hashes,
compiler notes, artifact hashes, and 9B-specific WGSL inventory are in the
workbench commit above.

## Exactness boundary

The fast path is enabled only for temperature-zero requests without
logprobs, grammar or structured output, logit bias, custom logit
processors, or frequency/presence/repetition penalties. Every other
request uses WebLLM's ordinary one-step sampler. K=4 may delay streaming
and cancellation by up to four tokens; speculative KV entries after a stop
are removed with scalar-typed KV PopN calls on both the paged attention
cache and the RNN state (the latter only survives this call if
`max_history_size` has been raised as described above).

Exactness here means deterministic greedy output equality on the recorded
ten-case corpus. It is not a claim of bit-identical intermediate logits or
stochastic-sampling parity.
