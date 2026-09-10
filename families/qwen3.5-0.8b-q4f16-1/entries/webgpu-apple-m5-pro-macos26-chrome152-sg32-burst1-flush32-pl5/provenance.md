# Qwen3.5-0.8B q4f16_1 WebGPU, prompt-lookup K=5 -- catalog ledger

Exact-greedy Qwen3.5-0.8B tuned on Apple M5 Pro / Chrome 152.0.7977.83 with
prompt-lookup drafting enabled. The runtime bundle source is
`OlehZhyhinas/web-llm-qwen@ff2ed7c61f5fd01b5fca8618cc2baa08c2866ad4`
(`prompt-lookup` branch, PR #1), based on
`OlehZhyhinas/web-llm-qwen@21698fd421d278ad0f7b3cc4e23abe3eaf4a042d`
(`qwen-m5`, source of the sibling burst5 bundle).

## Shipped stack

- WebLLM ABI 0.2.84 with prompt-lookup decoding configured by
  `promptLookup: "5:3:2:fork"` and loaded through
  `globalThis.__webllmPromptLookup`.
- MLC-LLM source `mlc-ai/mlc-llm@ed1c7f65f7fc4f08c53db1294c721f4c0ba49a35`.
- TVM/Relax source `mlc-ai/relax@c04f730addef05dda98f5ee23986de885ade9b9d`.
- Upstream model repository
  `mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC@0ec138972555613c1d7812a821778ad0398c8790`.
- Subgroup-32 model library and chunk-256 two-stage full-vocabulary GPU argmax,
  unchanged from the sibling burst5 entry.
- Runtime config at ship: burst1 + batchPass + flush32 + prompt-lookup K=5,
  with lookahead disabled.

This model is recurrent too (`mlc-chat-config.json` has
`full_attention_interval: 4`, and the model library exports
`create_rnn_state`). The older burst5 path could pop recurrent state without
enough history slots; prompt-lookup with `:fork` replaces that rollback path by
forking before verification and replaying only the accepted prefix on the fork.

## Methodology and measured result

Protocol: paired A/B within one Chrome session on 120 real arXiv paragraphs
(`LatexGen bench/arxiv-pastes.json`). Each item is generated once with drafting
off and once with drafting on, order alternating per item, `resetChat` between
sides, greedy decode, `max_tokens=512`, and `WEBNN_MAX_LOAD=0`.

### Row: qwen35-08b-arxiv-k5-shipped

- Baseline: previous entry knobs (`burst5 + batchPass + flush32`) with
  `max_history_size=8` so rollback had history slots.
- Median treatment/baseline ratio for ms per output character: **0.818**
  (IQR **0.740-0.907**), equivalent to **1.22x faster**.
- Median tokens/s ratio: **1.223**.
- Raw output byte-identical items: **119/120**.
- Tokens committed per decode pass: **2.96**.
- Passes with a draft: **58%**.
- Full acceptance among drafted passes: **67%**.
- Baseline decode: **6.1 ms/token**.
- Verify-pass cost at 6 drafted tokens: **C(6)=2.75** baseline decode steps.
- Divergent items judged with qwen-local: **1 judged**, **0 treatment-worse**,
  **1 both correct**, **0 both wrong**.

## Quality caveat

Output is **not** byte-identical in general. The `batch_verify` prefill kernels
and the decode GEMV kernels differ in low bits, so near-tie argmaxes flip on a
few percent of items.

Family byte-identical verification corpus result for this entry: **PENDING**.
