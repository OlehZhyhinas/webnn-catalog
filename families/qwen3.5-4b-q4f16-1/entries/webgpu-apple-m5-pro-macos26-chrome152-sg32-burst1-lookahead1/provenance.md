# Qwen3.5-4B q4f16_1 WebGPU, K=1 with GPU lookahead — catalog ledger

Exact-greedy Qwen3.5-4B tuned on Apple M5 Pro / Chrome 152.0.7977.77 -- the
model LatexGen ships today. The authoritative experiment record is
`OlehZhyhinas/webnn-workbench@1c7c5a62c6c74d796aeb5081cf0756ca900c68f4`,
`docs/qwen35-4b-m5-webgpu-report.md`. That commit is the head of workbench PR
[OlehZhyhinas/webnn-workbench#23](https://github.com/OlehZhyhinas/webnn-workbench/pull/23)
against `main`, not yet merged as of this writing.

## Shipped stack

- WebLLM 0.2.84 ABI, runtime source
  `OlehZhyhinas/web-llm-qwen@21698fd421d278ad0f7b3cc4e23abe3eaf4a042d`
  (branch `qwen-m5`; the lookahead-capable bundle, run here with lookahead
  enabled -- unlike the 8B and Qwen3.5-0.8B entries, which ship this same
  bundle with lookahead disabled).
- MLC-LLM source `mlc-ai/mlc-llm@ed1c7f65f7fc4f08c53db1294c721f4c0ba49a35`.
- TVM/Relax source `mlc-ai/relax@c04f730addef05dda98f5ee23986de885ade9b9d`,
  plus the isolated compiler-worktree fixes described in the workbench report.
- Upstream model repository
  `mlc-ai/Qwen3.5-4B-q4f16_1-MLC@44b42469f9e192814bfd90440e3b377d89ba7a13`.
- Subgroup-32 model library, chunk-256 two-stage full-vocabulary GPU argmax
  (WebGPU GEMV `TR` is a no-op for this model: byte-identical compiled output
  at every `TR` value 16/32/64/128, same finding as Qwen3.5-0.8B).
- Exact-greedy GPU-resident decode with a burst of one token and one
  speculative lookahead step kept queued on the GPU across each readback,
  no mid-burst flush cadence, no bind-group cache. Flush cadence and
  bind-group caching were both measured on top of this configuration and
  neither improved on it (see below).
- A `max_history_size` override (4: burst 1 + lookahead 1 + margin 2) on the
  model record so `vm.builtin.kv_state_popn` can correctly roll back this
  model's GatedDeltaNet recurrent/conv `RNNState` when the speculative
  lookahead step overshoots EOS. Published configs set `max_history_size=1`,
  which makes any real rollback throw; this override is the only difference
  from the published model configuration.

The selected model-library WASM is public at the Hugging Face dataset
revision recorded in `artifacts.json` and pinned again by byte count and
SHA-256 there. The runtime JS is byte-identical to, and reused from, the
existing Qwen3-1.7B lookahead catalog artifact at revision
`2efcfd5804a39bc2c1af40389e16d8a0a4e73d4a`, rather than duplicated; this was
confirmed by comparing SHA-256 before reuse
(`163177f1c2842ca60a52644109867c48e7ee6865530149a69ac3fd827d4cce35`).
Weights and tokenizer stay in the immutable upstream model revision. Catalog
runs used `context_window_size` 4096.

## Methodology and measured result

Every browser run used a separate real Google Chrome persistent context, the
dedicated `bench/.chrome-profile-qwen35-4b` profile, port 8912, and the
shared machine GPU lock. `WEBNN_MAX_LOAD=0` disabled load gating; the
machine's sustained background ORCA/NEB load remained active throughout.

The headline uses six rotating paired rounds, one fresh prompt per round.
The model and GPU pipelines are warm, but each measured observation calls
`resetChat(false)` and uses a different prompt, so this is not a
prompt-cache benchmark. Generation uses temperature zero and thinking
disabled. Ratios are computed within each round before taking the median.

- Published WebLLM 0.2.84 subgroup-32 median, stock runtime: **50.55 tokens/s**.
- Final stack median: **64.06 tokens/s** (**15.616 ms/token**).
- Median paired ratio: **1.276x**.
- Ratio IQR: 1.252-1.296.
- One-minute load-average range: 14.79-15.82 (headline run).
- Six of six fresh outputs, including completion-token counts, were
  byte-identical to the published path.
- Four of four quality outputs were byte-identical to local K=1
  full-vocabulary argmax.

There is no historical quiet baseline for this model. No quiet measurement
or projection is claimed.

## Model-specific tuning and negative results

Each retained choice was measured independently on this model:

- local compiler parity with GPU argmax disabled: 0.987x published on an
  initial four-round gate, 0.978x/0.967x for `TR=32`/`TR=64` on a second
  four-round set run alongside the TR sweep -- all three readings are
  consistent with each other within round-to-round noise on this shared,
  variably loaded machine, and none reversed direction, so no further
  compiler investigation was triggered;
- chunk-256 argmax alone was 0.992x fallback at K=1 (it pays off only once
  bursts are enabled, unlike Qwen3.5-0.8B); the full chunk sweep (64/128/
  256/512/1024) confirmed 256, with 128 and 512's nominal ties crossing
  parity on their own IQR;
- `TR` is a no-op for this model: every value 16/32/64/128 compiled to
  byte-identical WASM;
- K=8 beat K=1/4/5 on the sweep and on an eight-round K4/K5/K8 confirmation
  (1.043x K4, IQR clear of parity); K=5 confirmed below K4;
- flush 32 beat "ordinary" K=8 by 1.021x; flush 128's nominal 1.031x lead
  confirmed at 1.022x (IQR 1.009-1.032) over eight rounds -- a real but
  modest win inside the K=8 branch, made moot once lookahead superseded that
  branch entirely;
- bind-group caching was inconclusive (1.003x / 0.993x over two round
  counts, both IQRs crossing parity) -- not adopted, consistent with every
  earlier Qwen3/3.5 pass;
- **K=1 with GPU lookahead 1: 1.095x over the K=8/flush32 burst winner
  (IQR 1.082-1.108, clear of parity) -- adopted.** This clears the 1.05x
  adoption bar decisively, the same call the Qwen3-4B rung made at 1.063x,
  and the opposite of the 8B (1.043x) and Qwen3.5-0.8B (1.032x) rungs, both
  of which crossed or approached parity and were rejected;
- adding a mid-burst flush cadence back on top of the lookahead
  configuration did not help (1.002x, IQR crossing parity), so the shipped
  roll has no `flushEvery` at all.

The complete aggregate and per-round JSON, load averages, output hashes,
the GatedDeltaNet hybrid-cache caveat and fix, and the isolated WGSL
inventory are in the workbench commit above.

## Published-library divergence (not a tuned-stack issue)

The published `base` and `sg32` libraries were both measured and disagree
with each other on 2 of 6 fresh prompts (different completion-token counts,
a property of the two published libraries' differing fp16 accumulation
order). This entry's output tracks `sg32`, the baseline of record, byte for
byte on all 6 rounds, including the two rounds where `sg32` and `base`
disagree. `sg32` is 1.083x `base` on this machine (IQR 1.070-1.088).

## Exactness boundary

The fast path is enabled only for temperature-zero requests without logprobs,
grammar or structured output, logit bias, custom logit processors, or
frequency/presence/repetition penalties. Every other request uses WebLLM's
ordinary one-step sampler. The lookahead step past a stop is rolled back
with `vm.builtin.kv_state_popn`, which requires the `max_history_size`
override above to succeed on this hybrid model.

Exactness here means deterministic greedy output equality on the recorded
ten-case corpus. It is not a claim of bit-identical intermediate logits or
stochastic-sampling parity.
