# Qwen3.5-0.8B q4f16_1 WebGPU — catalog ledger

Exact-greedy Qwen3.5-0.8B tuned on Apple M5 Pro / Chrome 152.0.7977.77 -- the
model LatexGen ships today. The authoritative experiment record is
`OlehZhyhinas/webnn-workbench@63c47b91c5619b76443da81322ee78de168a087c`,
`docs/qwen35-08-m5-webgpu-report.md`.

## Shipped stack

- WebLLM 0.2.84 ABI, runtime source
  `OlehZhyhinas/web-llm-qwen@21698fd421d278ad0f7b3cc4e23abe3eaf4a042d`
  (branch `qwen-m5`; the lookahead-capable bundle, used with lookahead
  disabled for this entry).
- MLC-LLM source
  `mlc-ai/mlc-llm@ed1c7f65f7fc4f08c53db1294c721f4c0ba49a35`.
- TVM/Relax source
  `mlc-ai/relax@c04f730addef05dda98f5ee23986de885ade9b9d`,
  plus the isolated compiler-worktree fixes described in the workbench report.
- Upstream model repository
  `mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC@0ec138972555613c1d7812a821778ad0398c8790`.
- Subgroup-32 model library, chunk-256 two-stage full-vocabulary GPU argmax
  (WebGPU GEMV `TR` is a no-op for this model: byte-identical compiled
  output at every `TR` value 16/32/64/128/256), and K=5 GPU-resident
  exact-greedy decode.
- One WebGPU compute pass with a queue submit every 32 dispatches.
  Bind-group caching is disabled. Lookahead was measured and not adopted
  (see below); the shipped runtime bundle carries the lookahead pipeline
  but it is disabled for this entry (`runtime.config.lookahead=0`).
- A `max_history_size` override (7) on the model record so
  `vm.builtin.kv_state_popn` can correctly roll back this model's
  GatedDeltaNet recurrent/conv `RNNState` when a K=5 burst overshoots EOS.
  Published configs set `max_history_size=1`, which makes any real rollback
  throw; this override is the only difference from the published model
  configuration.

The selected model-library WASM is public at immutable Hugging Face dataset
revision `7c1591a3d2f4d4929f4d7afd6f07a17d598e15cf` and pinned again by byte
count and SHA-256 in `artifacts.json`. The runtime JS is byte-identical to,
and reused from, the existing Qwen3-1.7B lookahead catalog artifact at
revision `2efcfd5804a39bc2c1af40389e16d8a0a4e73d4a`, rather than duplicated;
this was confirmed by comparing SHA-256 before reuse
(`163177f1c2842ca60a52644109867c48e7ee6865530149a69ac3fd827d4cce35`).
Weights and tokenizer stay in the immutable upstream model revision. Catalog
runs used `context_window_size` 4096.

## Methodology and measured result

Every browser run used a separate real Google Chrome persistent context, the
dedicated `bench/.chrome-profile-qwen35-08` profile, port 8910, and the
shared machine GPU lock. ORCA continued saturating the CPU; `WEBNN_MAX_LOAD=0`
disabled load gating rather than stopping or starving it.

The headline uses eight rotating paired rounds (six distinct prompts, two
repeated). The model and GPU pipelines are warm, but each measured
observation calls `resetChat(false)` and uses a different prompt, so this is
not a prompt-cache benchmark. Generation uses temperature zero and thinking
disabled. Ratios are computed within each round before taking the median.

- Published WebLLM 0.2.84 subgroup-32 median, stock runtime: **108.88 tokens/s**.
- Final stack median: **190.83 tokens/s** (**5.24 ms/token**).
- Median paired ratio: **1.738x**.
- Ratio IQR: 1.705-1.769.
- One-minute load-average range: 14.4-15.22 (headline run).
- Eight of eight fresh outputs, including completion-token counts, were
  byte-identical to the published path.
- Four of four quality outputs were byte-identical to local K=1
  full-vocabulary argmax *and* to the published path directly -- no
  published-sampler instability was observed on this corpus for this model.

There is no historical quiet baseline for this model. No quiet measurement
or projection is claimed.

## Model-specific tuning and negative results

The 8B stack was used only as a starting point; each retained choice was
remeasured on this model, which behaved differently on several axes:

- local compiler parity with GPU argmax disabled: 0.987x published (8-round
  confirmation), after an initial 0.967x run that triggered the parity gate;
- chunk-256 GPU argmax was already a standalone 1.039x K=1 speedup (unlike
  the larger Qwen3 models, where argmax only paid off once bursts were
  enabled);
- chunks 64/512/1024 did not beat 256; chunk128's nominal tie crossed parity
  on 8-round confirmation;
- `TR` is a no-op for this model: every value 16/32/64/128/256 compiled to
  byte-identical WASM, so no `TR` sweep was run beyond confirming this;
- K=5 beat K=4 and K=8 on both the sweep and an 8-round K4/K5/K8
  confirmation (1.033x K4) -- unlike the 8B pass, where K=4 won outright;
- flush 32: 1.072x ordinary K=5; flush 16/64 measured lower; flush 128's
  nominal 1.077x lead reversed to 1.017x flush32 with an IQR crossing
  parity on 8-round confirmation;
- bind-group caching: inconclusive 1.003x with an IQR crossing parity;
- K=1 with GPU lookahead 1: 1.032x this entry, IQR crossing parity, below
  the 1.05x adoption bar -- not adopted, the same call as the 8B pass.

This model's overall gain (1.738x) is the largest of the five Qwen3/3.5
catalog rolls: it is the smallest, most overhead-bound rung, so burst+flush
tuning removes proportionally more of the fixed per-decode-step cost, and
correspondingly less headroom is left for lookahead to hide.

The complete aggregate and per-round JSON, load averages, output hashes,
the GatedDeltaNet hybrid-cache caveat and fix, and the isolated WGSL
inventory are in the workbench commit above.

## Exactness boundary

The fast path is enabled only for temperature-zero requests without logprobs,
grammar or structured output, logit bias, custom logit processors, or
frequency/presence/repetition penalties. Every other request uses WebLLM's
ordinary one-step sampler. K=5 may delay streaming and cancellation by up to
five tokens; speculative KV and recurrent-state entries after a stop are
removed with `vm.builtin.kv_state_popn`, which requires the
`max_history_size` override above to succeed on this hybrid model.

Exactness here means deterministic greedy output equality on the recorded
ten-case corpus. It is not a claim of bit-identical intermediate logits or
stochastic-sampling parity.
