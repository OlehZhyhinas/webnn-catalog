# Texify (Donut-Swin + mBART, ~300M) as hand-built WebNN graphs

Working notes for [issue #3](https://github.com/OlehZhyhinas/webnn-workbench/issues/3).
Machine and protocol as in [scope.md](scope.md): Apple M5 Pro, Chrome 152,
WebNN on Core ML (`preferredInputLayout === "nchw"`), persistent profile
`bench/.chrome-profile-texify`, HTTP port **8905**.

Everything lives under `bench/webnn/texify/`. Do not use the in-IDE browser;
Playwright launches real Google Chrome (`chromium.launchPersistentContext`,
`channel: "chrome"`).

## Status

| | |
|---|---|
| Weights | `bench/webnn/texify/weights/` (626 MB fp16, 592 tensors, 312.4M params) |
| fp32 refs | `bench/webnn/texify/reference/` (18 images, encoder K/V, decode step 0, greedy ids) |
| fp16 audit | [bench/results/texify-fp16-range.md](../bench/results/texify-fp16-range.md) |
| fp16 floor (MPS, first image) | encoder max abs 7.7e-2; decode step 0 logits 9.1e-3, token match |
| Encoder | 1049 WebNN ops vs 16,003 ORT nodes; K/V within 2× floor on 18/18 |
| Greedy | **18/18** vs fp32 Hugging Face; warm median **146 ms** (bar 400 ms) |
| Catalog | `families/texify-420/` (issue path `models/texify-420/`) |

## The model, as LatexGen ships it

Read off `vikp/texify` (fp32) and `LatexGen/public/models/texify/config.json`.

- **Encoder** (Donut-Swin): 420×420 RGB, patch 4 → 105×105×128. Depths
  `[2,2,14,2]`, heads `[4,8,16,32]`, window 5, GELU, qkv bias, no absolute
  position embeddings, no final encoder layer-norm. Live spatial sizes after
  odd-side patch-merging are **105 → 53 → 27 → 14**, not the constructor's
  `105//2^i`. Window pad is to the next multiple of 5: **53→55, 27→30, 14→15**.
  Output `last_hidden_state` `[1,196,1024]`.
- **Decoder** (mBART): 8 layers, d_model 1024, 16 heads × 64, FFN 4096, exact-erf
  GELU, learned positions with offset 2 (table `[1538,1024]`),
  `scale_embedding` (×32), `layernorm_embedding`, final `layer_norm`,
  untied `lm_head` `[50000,1024]`. Greedy, start/bos 0, eos 2, pad 1,
  `forced_eos_token_id` 2, `max_new_tokens` 384. Cross-attention over 196
  encoder tokens; no `enc_to_dec_proj` (hidden sizes already match).
- **Preprocessing** stays in the consumer (thumbnail + pad). The export spells
  the transformers 4.36 Donut pipeline (shortest-edge resize BILINEAR, thumbnail
  BICUBIC, center pad 0, /255, ImageNet mean/std) because transformers 5's
  `DonutImageProcessor` dropped thumbnail+pad and returned the raw screenshot
  size.

## fp16 range audit (18 images)

No activation, linear, conv, or softmax *score* overflows fp16 (tightest
activation margin ~40× on LayerNorm inputs, ~100× on linears, ~500× on
softmax). Several LayerNorm **sum-of-squares** accumulators would overflow a
naive fp16 `sum(x^2)` — tightest 3.8e6 at stage-2 patch-merge LN, margin
0.017×. Variance `mean(x^2)` still has >18×. **Remedy: WebNN
`layerNormalization`** (Core ML uses a higher-precision accumulator), the same
shape of fix as IntelliTeX's fp32 RMSNorm variance. Do not hand-roll LayerNorm
as fp16 sum-then-divide. No per-layer rescale and no FFN/softmax fp32 islands.

This is the plausible cause of LatexGen's recorded "fp16 on WebGPU produced
garbage" under ORT Web: a generic fp16 LN kernel accumulating `sum(x^2)` Infs
at Donut-Swin stage 2 (512-dim, values ~1e3).

## Graphs

- **Encoder:** `[1,3,420,420]` fp16 NCHW in; hoisted decoder EncDecAttention
  K/V out, 8 × `[1,16,196,64]`. Window partition as rank-5 reshape/transpose
  (Chrome 152 rejects rank 6); cyclic shift as slice+concat (gather is
  `--shift-style gather`); packed QKV; `1/sqrt(32)` folded into encoder Q
  (every stage has head dim 32). 1049 ops, 174 MiB, ~6 s compile.
- **Decode-step:** token + step int32 `[1]`; learned pos gathered as
  `step+2`; embed scale folded into the token table; one-hot cache ping-pong;
  causal mask row gathered by step; argMax in-graph; logits only on
  `--logits-out`. Cache shape `[1,16,384,64]`. 349 ops, 423 MiB, ~9 s compile.

## Reproduce

```bash
# from a webnn-workbench checkout, branch texify
.venv/bin/python bench/webnn/texify/export_texify.py
node bench/webnn/texify/run.mjs --stage encoder --label smoke --images pythagorean --port 8905
node bench/webnn/texify/run.mjs --stage e2e --label greedy-18 --port 8905
node bench/webnn/texify/make-recipes.mjs
```
