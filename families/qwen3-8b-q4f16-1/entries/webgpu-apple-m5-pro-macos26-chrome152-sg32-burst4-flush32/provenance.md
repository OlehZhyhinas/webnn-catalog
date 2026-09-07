# Qwen3-8B q4f16_1 WebGPU — catalog ledger

Exact-greedy Qwen3-8B tuned on Apple M5 Pro / Chrome 152.0.7977.77.
The authoritative experiment record is
`OlehZhyhinas/webnn-workbench@045f6f00a86e436416644d98cb13d8aa5b6d8c92`,
`docs/qwen8-m5-webgpu-report.md`.

## Shipped stack

- WebLLM 0.2.84 ABI, runtime source
  `OlehZhyhinas/web-llm-qwen@b9e2b01d3d3f4b5e1fad005da159e31430c65130`.
- MLC-LLM source
  `mlc-ai/mlc-llm@ed1c7f65f7fc4f08c53db1294c721f4c0ba49a35`.
- TVM/Relax source
  `mlc-ai/relax@c04f730addef05dda98f5ee23986de885ade9b9d`,
  plus the isolated compiler-worktree fixes described in the workbench report.
- Upstream model repository
  `mlc-ai/Qwen3-8B-q4f16_1-MLC@b3d55c289eae58f77095f5b68c895eeea358ee09`.
- Subgroup-32 model library, chunk-256 two-stage full-vocabulary argmax,
  GEMV `TR=32`, and K=4 GPU-resident exact-greedy decode.
- One WebGPU compute pass with a queue submit every 32 dispatches.
  Bind-group caching is disabled. Lookahead is not part of this headline.

The selected model-library WASM is public at immutable Hugging Face dataset
revision `baf469ccf9a29720282d6bee0c9d743436cdc9af` and pinned again by byte
count and SHA-256 in `artifacts.json`. The unchanged runtime JS is reused from
the existing 1.7B artifact at revision `170ebdf3b1707f939833511cf1d295305227c921`
rather than duplicated. Weights and tokenizer stay in the immutable upstream
model revision. Catalog runs used `context_window_size` 4096.

## Methodology and measured result

Every browser run used a separate real Google Chrome persistent context, the
dedicated `bench/.chrome-profile-qwen8` profile, port 8909, and the shared
machine GPU lock. ORCA continued saturating the CPU; `WEBNN_MAX_LOAD=0`
disabled load gating rather than stopping or starving it.

The headline uses six rotating paired rounds. The model and GPU pipelines are
warm, but each measured observation calls `resetChat(false)` and uses a
different prompt, so this is not a prompt-cache benchmark. Generation uses
temperature zero and thinking disabled. Ratios are computed within each round
before taking the median.

- Published WebLLM 0.2.84 subgroup-32 median: **39.22 tokens/s**.
- Final stack median: **46.19 tokens/s** (**21.65 ms/token**).
- Median paired ratio: **1.187x**.
- Ratio IQR: 1.154–1.216; MAD 0.035; range 1.139–1.222.
- One-minute load-average range: 16.09–18.46.
- Six of six fresh outputs, including completion-token counts, were
  byte-identical to the published path.
- Four of four quality outputs were byte-identical to local K=1
  full-vocabulary argmax. The published temperature-zero sampler alternated
  on the math item, so that item is not claimed as a published-path match.

There is no historical quiet 8B number. No quiet measurement or projection is
claimed.

## Model-specific tuning and negative results

The 4B stack was used only as a starting point; each retained choice was
remeasured on 8B:

- local compiler parity with GPU argmax disabled: 0.991x published `TR=32`,
  0.989x default `TR=64`, after an initial 0.971x run that triggered the
  parity gate;
- chunk-256 GPU argmax: 0.976x fallback, kept because it enabled the burst
  win;
- chunk 512 initially appeared 1.010x faster, then reversed to 0.992x over
  eight confirmation rounds;
- chunks 64/128/1024 did not beat 256;
- `TR=64` initially measured 1.023x `TR=32`, then reversed to 0.995x;
- `TR=16/128/256`: 0.801x/0.995x/0.944x `TR=32`;
- K=4: 1.122x K=1;
- K=5 and K=8 reversed to 0.990x/0.995x K=4;
- flush 32: 1.066x ordinary K=4;
- flush 16: 0.988x flush 32;
- flush 64/128: 0.995x/1.006x flush 32, not retained;
- bind-group caching: inconclusive 1.004x.

The GPU profile was dominated by quantized GEMV. It did not show a compelling
host/embed seam, so a fused entrypoint was not added. Lookahead was not mixed
into this first 8B headline.

The complete aggregate and per-round JSON, load averages, output hashes,
compiler notes, context scaling, artifact hashes, and 8B-specific WGSL
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
