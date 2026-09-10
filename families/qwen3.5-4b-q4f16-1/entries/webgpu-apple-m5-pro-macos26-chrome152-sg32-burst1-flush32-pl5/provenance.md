# Qwen3.5-4B q4f16_1 WebGPU, prompt-lookup K=5 -- catalog ledger

Exact-greedy Qwen3.5-4B tuned on Apple M5 Pro / Chrome 152.0.7977.83 with
prompt-lookup drafting enabled. The runtime bundle source is
`OlehZhyhinas/web-llm-qwen@ff2ed7c61f5fd01b5fca8618cc2baa08c2866ad4`
(`prompt-lookup` branch, PR #1), based on
`OlehZhyhinas/web-llm-qwen@21698fd421d278ad0f7b3cc4e23abe3eaf4a042d`
(`qwen-m5`, source of the sibling lookahead bundle).

## Shipped stack

- WebLLM ABI 0.2.84 with prompt-lookup decoding configured by
  `promptLookup: "5:3:2:fork"` and loaded through
  `globalThis.__webllmPromptLookup`.
- MLC-LLM source `mlc-ai/mlc-llm@ed1c7f65f7fc4f08c53db1294c721f4c0ba49a35`.
- TVM/Relax source `mlc-ai/relax@c04f730addef05dda98f5ee23986de885ade9b9d`.
- Upstream model repository
  `mlc-ai/Qwen3.5-4B-q4f16_1-MLC@44b42469f9e192814bfd90440e3b377d89ba7a13`.
- Subgroup-32 model library and chunk-256 two-stage full-vocabulary GPU argmax,
  unchanged from the sibling lookahead entry.
- Runtime config at ship: burst1 + batchPass + flush32 + prompt-lookup K=5,
  with lookahead disabled.

The runtime JS and model WASM are hash-pinned in `artifacts.json`.

## Methodology and measured result

Protocol for all rows: paired A/B within one Chrome session on 120 real arXiv
paragraphs (`LatexGen bench/arxiv-pastes.json`). Each item is generated once
with drafting off and once with drafting on, order alternating per item,
`resetChat` between sides, greedy decode, `max_tokens=512`, and
`WEBNN_MAX_LOAD=0`.

### Row: qwen35-4b-arxiv-k5-shipped

- Baseline: this entry's other knobs with prompt-lookup disabled
  (`burst1 + batchPass + flush32`).
- Median treatment/baseline ratio for ms per output character: **0.863**
  (IQR **0.790-0.954**), equivalent to **1.16x faster**.
- Median tokens/s ratio: **1.159**.
- Raw output byte-identical items: **116/120**.
- Tokens committed per decode pass: **2.23**.
- Passes with a draft: **40%**.
- Full acceptance among drafted passes: **47%**.
- Baseline decode: **22.2 ms/token**.
- Verify-pass cost at 6 drafted tokens: **C(6)=2.63** baseline decode steps.
- Divergent items judged with qwen-local (three repeats each): **4 judged**,
  **0 treatment-worse**, **1 both correct**, **3 both wrong**.

### Row: qwen35-4b-arxiv-k5

- Baseline: plain single-step decode (prompt-lookup off, no batchPass, no
  flush cadence).
- Median treatment/baseline ratio for ms per output character: **0.789**
  (IQR **0.727-0.882**).
- Raw output byte-identical items: **119/120**.
- Baseline decode: **18.2 ms/token**.
- Verify-pass cost at 6 drafted tokens: **C(6)=2.29** baseline decode steps.

## Dropped points

- `qwen35-4b-arxiv-k10`: **0.905** (IQR **0.819-1.006**) with IQR touching
  parity, raw output byte-identical **113/120**, and
  **C(11)=4.02** baseline decode steps.
- Rendered `pdf-pastes` development corpus:
  - `k=10`: **1.18** (IQR **1.11-1.26**) -- slower than baseline.
  - `k=5`: **1.02** (IQR **0.92-1.09**) -- neutral.
- Decision: keep `k=5`; drop `k=10`.

## Quality caveat

Output is **not** byte-identical in general. The `batch_verify` prefill kernels
and the decode GEMV kernels differ in low bits, so near-tie argmaxes flip on a
few percent of items.

Family byte-identical verification corpus result for this entry: **10/10 corpus cases byte-identical, verified 2026-09-10 with scripts/verify-webllm.mjs against this branch's raw entry URL (bench/verify-qwen3.5-4b-q4f16-1-pl5.json)**.
