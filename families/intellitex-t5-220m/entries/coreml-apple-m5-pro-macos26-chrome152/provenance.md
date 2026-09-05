# IntelliTeX (CodeT5+ 220M T5) — catalog ledger

Hand-built WebNN encoder + decode-step graphs for `duanxianpi/IntelliTex`,
measured 2026-09-05 on Apple M5 Pro / Chrome 152 Core ML
(`preferredInputLayout === "nchw"`). Workbench commit `97140a1`.

## What shipped

Three static encoder graphs (buckets 32 / 64 / 128) and three decode-step
graphs (cache length 256). Encoder emits the 12 decoder layers' EncDecAttention
K/V; decode takes one token + step + 24 self-attention caches + those K/V,
writes the updated caches, and returns `argMax` as int32 `[1]`.

Warm median **146.2 ms** on the 15 LatexGen items, **15/15** greedy tokens
identical to fp32 Hugging Face `generate()` (leading `decoder_start` 0
stripped). Isolated encoder ~5 ms/bucket; isolated decode step 0 ~5.2 ms.

LatexGen today: 450 ms ORT Web WebGPU int4, 1280 ms ORT CPU int8.

## Folds

- Packed QKV; relative-bias tables baked per encoder length.
- Decoder EncDecAttention K/V hoisted into the encoder.
- Tied lm_head = embedding scaled by `d_model^-0.5`.
- RMSNorm variance in fp32: a naive fp16 sum-of-squares of `1403 * 768`
  overflows. fp16 range audit: 0 flagged / 451 sites; bar 3.8e-2.

## Pass-one bug (do not regress)

The baked decoder table is Hugging Face `compute_bias` layout
`[1, heads, query, key]`. A reshape to `[256, 12, 256]` is not a transpose
— heads stay the slowest axis — so `gather(axis=0)` read the wrong query
row. Step 0 cannot see it: causal softmax over one key is identically 1.
Gathering **axis 2** fixes every later step. Relative bias is still added on
all 12 decoder layers; only layer 0 owns the embedding, matching HF.

A JS round-trip of the 24 cache tensors was chasing that symptom. On-device
A/B ping-pong is correct.

## Recording notes

- Compiling three decode graphs in one WebNN context kills Chrome. The
  workbench and `scripts/verify.mjs` build one bucket per fresh page.
- Decode recipes rename outputs (`token_out`, `selfN_k_out`) so input and
  output names differ. The recorded operand ids are unchanged.
- Timed e2e does not emit logits (~64 KB/step of unused readback).
- IR dumps: `bench/webnn/ir/intellitex-{encoder,decode}-L{32,64,128}.{json,bin}`.
