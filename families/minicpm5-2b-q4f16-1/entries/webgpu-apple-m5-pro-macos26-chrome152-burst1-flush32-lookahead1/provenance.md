# MiniCPM5-2B q4f16_1 WebGPU — catalog ledger

Exact-greedy MiniCPM5-2B tuned on Apple M5 Pro / Chrome 152.0.7977.77.
The authoritative experiment record is
`OlehZhyhinas/webnn-workbench@minicpm5-2b-roll`,
`docs/minicpm5-2b-m5-webgpu-report.md`.

## Shipped stack

- WebLLM 0.2.84 ABI, runtime source
  `OlehZhyhinas/web-llm-qwen@21698fd421d278ad0f7b3cc4e23abe3eaf4a042d`
  (the 1.7B/4B/8B lookahead bundle, reused unchanged).
- MLC-LLM source
  `mlc-ai/mlc-llm@ed1c7f65f7fc4f08c53db1294c721f4c0ba49a35`.
- TVM/Relax source
  `mlc-ai/relax@c04f730addef05dda98f5ee23986de885ade9b9d`,
  plus the isolated compiler-worktree dlight fixes described in the
  workbench report (the same `get_sblock` fix used for the Qwen3 rolls).
- Upstream model repository
  `ozhyhinas/MiniCPM5-2B-q4f16_1-MLC@2318f37d9c39277ff01dc64086491028c95d4db4`
  (Oleh's own q4f16_1 quantization of `openbmb/MiniCPM5-2B`).
- **No WebGPU subgroups** (unlike every Qwen3 catalog entry), chunk-256
  two-stage full-vocabulary GPU argmax, and GEMV `TR=64` (TVM's unmodified
  default, not retuned). Subgroups and `TR=32` each independently break
  exact-greedy parity on this model: subgroup GEMV reduction changes the
  accumulation order enough to flip a near-tie logit on one of six tested
  fresh prompts. This is the same class of cosmetic base-vs-sg32 divergence
  already documented for the Qwen3 catalog entries (identical LaTeX,
  unequal token count on the flipped prompt), not a MiniCPM5-2B-specific
  defect; there is no `mlc-ai`/subgroup-32 publication for this model, so
  exactness was defined against the only published (no-subgroup) library.
- Exact-greedy GPU-resident decode with a burst of one token and one
  decode step kept queued on the GPU across each readback (`K=1` + GPU
  lookahead 1), adopted after a second pass measured against a plain
  `K=4` burst.
- One WebGPU compute pass with a queue submit every 32 dispatches.
  Bind-group caching is disabled.

The selected model-library WASM is public at immutable Hugging Face dataset
revision `3ee998cdd86fd43ac417edebab9b16aaba03e700` and pinned again by byte
count and SHA-256 in `artifacts.json`. The runtime JS is reused from the
existing 1.7B lookahead artifact at revision
`2efcfd5804a39bc2c1af40389e16d8a0a4e73d4a` rather than duplicated. Weights,
tokenizer and the published (stock) model library stay in the immutable
upstream model revision. The published baseline library's SHA-256 is
`ef11c667aec8fa65525aaa0a3e11c9f9dedb18d513162a07a111accca4bd84e9`
(`MiniCPM5-2B-q4f16_1-MLC-webgpu.wasm`). Catalog runs used
`context_window_size` 4096.

## Methodology and measured result

Every browser run used a separate real Google Chrome persistent context, a
dedicated Chrome profile, a dedicated port, and the shared machine GPU lock.
Other model-roll benchmarks shared the machine concurrently;
`WEBNN_MAX_LOAD=0` disabled load gating rather than stopping or starving
that work. There is no `mlc-ai`-published library and no subgroup-32
variant for this model, so the published self-compiled library is the only
baseline of record.

The headline uses eight rotating paired rounds, live against the published
path. The model and GPU pipelines are warm, but each measured observation
calls `resetChat(false)` and uses a different prompt, so this is not a
prompt-cache benchmark. Generation uses temperature zero. Ratios are
computed within each round before taking the median.

- Published WebLLM 0.2.84 median: **74.885 tokens/s**.
- Pre-lookahead K=4/flush-32 stack median: **95.73 tokens/s**, 1.273x
  published, 8/8 byte-identical.
- Final stack (this entry) median: **102.665 tokens/s**
  (**9.74 ms/token**), **1.362x** published.
- Ratio IQR: 1.357-1.378; range 1.320-1.406.
- One-minute load-average range: 15.04-17.6.
- Eight of eight fresh outputs, including completion-token counts, were
  byte-identical to the published path, for both the K=4 stack and this
  entry, in every round.
- Four of four quality outputs were byte-identical to a local plain-K=1
  reference, over four rounds. No published-sampler nondeterminism was
  observed for this model on any tested prompt.

Paired end-to-end (TTFT + decode) against the same published baseline: the
K=4 stack measured 1.229x median, this entry 1.281x median, and this
entry's end-to-end ratio exceeded the K=4 stack's in every one of the
8 rounds — the decisive evidence for the lookahead adoption below.

There is no historical quiet MiniCPM5-2B number. No quiet measurement or
projection is claimed.

## GPU lookahead second pass and adoption decision

A second pass paired `K=1` + GPU lookahead 1 live against the K=4/flush-32
stack, same model library and runtime bundle: **1.068x** decode (IQR
1.052-1.093, range 0.992-1.179, n=8, byte-identical on 8/8 fresh and 4x4
quality-corpus outputs). This clears the 1.05x adoption bar the 8B pass's
1.043x missed. The next-request TTFT cost is real: **+7.7 ms average**
(range +7.3 to +8.3 ms), the same orphaned-speculative-step mechanism
documented for the 8B pass, but smaller in absolute terms because a
MiniCPM5-2B decode step (~10.4 ms) is itself much shorter than 8B's
(~19 ms). Paired end-to-end stays net positive on both the fresh corpus
(1.031x, IQR 1.025-1.051) and the quality corpus (1.022x, IQR
1.017-1.029), unlike the 8B pass's flat 1.006x fresh-corpus result.

**Decision: adopted.** The decisive evidence is the direct
published-path comparison above: paired live in the same rotation, this
entry's end-to-end ratio against published beat the K=4 stack's in every
one of 8 rounds. `K=1` + lookahead 1 + flush 32 is this entry's
configuration; the K=4/flush-32 numbers are kept as the pre-lookahead
baseline the decision is measured against, not a separate catalog entry.

## Model-specific tuning and negative results

The Qwen3 pattern (subgroups enabled, retune `TR`) was tried first and
**failed the exactness bar**, not the throughput bar:

- subgroups + `TR=32`/`TR=64`, argmax disabled: 1.200x/1.190x published,
  both **diverged from the published text** on one of six fresh prompts;
- no subgroups + `TR=32`: still diverged (1.004x, not exact);
- no subgroups + `TR=64` (TVM default): byte-identical, 0.973x published
  (compiler-tree/load noise) — the parity gate pass;
- `TR=16/128`: 0.857x/0.908x, too slow;
- chunk-256 GPU argmax: 1.021x fallback, kept because it enabled the burst
  win;
- chunk 512's apparent 1.029x-class lead reversed to 0.995x over eight
  confirmation rounds; 64/128/1024 did not displace 256;
- K=4: 1.206x K=1 in the sweep, held 1.000x (reference) on an eight-round
  confirmation where K5 reversed to 0.996x (IQR crossing parity) and K8
  measured 0.951x;
- flush 16/32/64/128: statistically tied; flush 32 kept by convention,
  matching every Qwen3 size;
- bind-group caching: inconclusive 1.002x.

The complete aggregate and per-round JSON, load averages, output hashes,
compiler notes, artifact hashes, and MiniCPM5-2B-specific WGSL inventory
are in the workbench commit above.

## Exactness boundary

The fast path is enabled only for temperature-zero requests without
logprobs, grammar or structured output, logit bias, custom logit
processors, or frequency/presence/repetition penalties. Every other
request uses WebLLM's ordinary one-step sampler. `K=1` with lookahead 1
keeps one decode step queued on the GPU across each readback; on stop, the
in-flight step's KV entries are removed with the normal KV PopN path, and
the orphaned step's remaining GPU work is what costs the next request's
TTFT (see the workbench report for the mechanism).

Exactness here means deterministic greedy output equality on the recorded
ten-case corpus. It is not a claim of bit-identical intermediate logits or
stochastic-sampling parity.
