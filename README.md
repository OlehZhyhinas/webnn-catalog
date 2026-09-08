# webnn-catalog

Pre-tuned browser inference, published as data. You point a loader at an entry
and get a running model in the page. There's no build step and nothing to
install natively.

For scale: SD-Turbo generates a 512×512 image from a new prompt in **68 ms** on
an M5 Pro in Chrome. ONNX Runtime Web on WebGPU takes 916 ms for the same model
on the same machine, and native PyTorch on MPS takes 94.5 ms. The runtime is the
WebNN and WebGPU that Chrome already ships. What's different is the order the
graph is built in, which someone worked out by hand and then recorded.

There is no npm package. `runtime/` holds two ES modules with no dependencies.
Copy them into your project or serve them from wherever you serve static files.

---

## What you can run

### Image and LaTeX: WebNN / Core ML

| family | task | in → out | download | new prompt |
|---|---|---|---:|---:|
| [`sd-turbo-512-1step`](families/sd-turbo-512-1step/) | text → image | prompt → 512×512 RGBA | 2.41 GB | **68.4 ms** |
| [`texo-384`](families/texo-384/) | image → LaTeX | 384×384 grayscale crop → LaTeX | 44 MB | **25.9 ms** |
| [`texify-420`](families/texify-420/) | image → LaTeX | 420×420 RGB crop → LaTeX | 627 MB | **145.7 ms** |
| [`intellitex-t5-220m`](families/intellitex-t5-220m/) | text → LaTeX | English → LaTeX | 1.04 GB | **146.2 ms** |

Against the fp32 model each one was built from: SD-Turbo comes out at PSNR
44.28 dB. Texo and Texify produce identical greedy tokens on 18/18 benchmark
images. IntelliTeX produces identical tokens to fp32 Hugging Face on 15/15 items.

If you want a comparison for Texo, LatexGen's own ONNX Runtime Web path takes
770 ms per image on WASM fp32 and 780 ms on WebGPU. This catalog's entry takes
25.9 ms.

### Chat: WebLLM / WebGPU

OpenAI-compatible chat with exact greedy decode. Most families have two
entries; the 8B, Qwen3.5-0.8B, MiniCPM5-2B and Qwen3.5-9B each have one.
Entries within a family share a model library and differ in how decode is
tuned.

| family | entry | download | throughput | vs stock WebLLM |
|---|---|---:|---:|---:|
| [`qwen3-0.6b-q4f16-1`](families/qwen3-0.6b-q4f16-1/) | `…-sg32-burst4` | 359 MB | 250.7 tok/s | 1.504× |
| | `…-sg32-burst4-flush64` | 359 MB | **312.6 tok/s** | 1.934× |
| [`qwen3-1.7b-q4f16-1`](families/qwen3-1.7b-q4f16-1/) | `…-sg32-burst4-flush32` | 996 MB | 139.9 tok/s | 1.371× |
| | `…-burst1-flush32-lookahead1` | 996 MB | **158.7 tok/s** | 1.587× |
| [`qwen3-4b-q4f16-1`](families/qwen3-4b-q4f16-1/) | `…-sg32-burst4-flush32` | 2.29 GB | 73.9 tok/s | 1.285× |
| | `…-burst1-flush32-lookahead1` | 2.29 GB | **78.5 tok/s** | 1.332× |
| [`qwen3-8b-q4f16-1`](families/qwen3-8b-q4f16-1/) | `…-sg32-burst4-flush32` | 4.64 GB | **46.2 tok/s** | 1.187× |
| [`qwen3.5-0.8b-q4f16-1`](families/qwen3.5-0.8b-q4f16-1/) | `…-sg32-burst5-flush32` | 460 MB | **190.8 tok/s** | 1.738× |
| [`minicpm5-2b-q4f16-1`](families/minicpm5-2b-q4f16-1/) | `…-burst1-flush32-lookahead1` | 1.45 GB | **102.7 tok/s** | 1.362× |
| [`qwen3.5-9b-q4f16-1`](families/qwen3.5-9b-q4f16-1/) | `…-sg32-burst4-flush64` | 5.07 GB | **37.45 tok/s** | 1.070× |

The last column is a paired, interleaved measurement against the published
WebLLM 0.2.84 subgroup-32 path on the same machine, with byte-identical output
required. For the 1.7B and 4B `flush32` entries, the 8B, the Qwen3.5-0.8B, the
MiniCPM5-2B and the Qwen3.5-9B, throughput and ratio come from the same run.
For the other four, the throughput is the entry's recorded headline and the
ratio comes from a later paired run; each entry's `measurements.json` has
both. The `lookahead1` entries stream one token at a time; the `burst4`
entries produce tokens in groups of four.

All of these numbers come from one machine: Apple M5 Pro, macOS 26.6.2, Chrome
152. Each entry directory has a `measurements.json` and a `provenance.md` that
give the baseline, protocol, host, and the conditions the run was done under.
If you're going to quote a figure, look at those first.

There's a machine-readable index in [`catalog.json`](catalog.json).

## What you need

- **Chrome 152 or newer.**
- **For WebNN entries:** WebNN enabled, and a **non-incognito profile**.
  Chromium only enables the Core ML backend for a real on-disk profile. In an
  off-the-record profile it falls back to CPU without any error, and runs
  roughly 50× slower. `assertCoreMLFingerprint(ctx)` detects this and throws.
- **For WebLLM entries:** WebGPU with `shader-f16` and subgroup size 32. The
  loader checks for these before downloading anything.
- **Disk and bandwidth for the download.** Sizes are in the tables above. WebNN
  constants are streamed in chunks, so resident memory during load stays around
  64–98 MiB no matter how big the blob is.

Every entry here was tuned on Apple silicon. `entry.compat.requires` lists the
conditions that have to hold for an entry to work at all, and that's the field
to filter on. See [choosing an entry](#choosing-an-entry).

## Using it

There are two loaders. Which one you use depends on the entry's `runtimeKind`.

### Chat: `runtimeKind: "webllm"`

```js
import { loadWebLLMFromUrl } from "./runtime/webllm-loader.js";

const rig = await loadWebLLMFromUrl(entryUrl);

const out = await rig.engine.chat.completions.create({
  messages: [{ role: "user", content: "Explain WebGPU subgroups briefly." }],
  temperature: 0,
  max_tokens: 256,
});
console.log(out.choices[0].message.content);

await rig.dispose();
```

`loadWebLLMFromUrl()` checks for subgroup-32 support, fetches the entry's
artifact manifest, and verifies the byte count and SHA-256 of the runtime
JavaScript and the model-library WASM. Only after that does it expose them, via
temporary blob URLs. Model weights are fetched from an upstream Hugging Face
repository at a pinned revision; the catalog doesn't host copies. `dispose()`
unloads the engine and revokes the blob URLs.

The exact-greedy fast path is used when `temperature` is 0 and none of the
following are set: logprobs, penalties, logit bias, grammar, structured output,
or a custom logit processor. Any other request goes through WebLLM's normal
sampler with the same API, and you won't notice the difference from the
outside. `isGreedyBurstEligible()` implements that check if you want to run it
yourself.

### Image and LaTeX: `runtimeKind: "webnn"`

```js
import {
  loadEntry, createEntryTensors, assertCoreMLFingerprint,
} from "./runtime/loader.js";

const entry = await (await fetch(`${dir}/entry.json`)).json();

const ctx = await navigator.ml.createContext({ deviceType: "gpu" });
assertCoreMLFingerprint(ctx);   // throws if the backend fell back to CPU

// null: fetch constants from where the manifest publishes them, in the chunks
// it names. A base URL ("/weights") points at a local mount instead.
const rig = await loadEntry(entry, null, ctx, { baseUrl: dir });
const t   = await createEntryTensors(ctx, rig);

ctx.writeTensor(t.get("text", "input_ids"), ids);   // 77 CLIP ids, int32
ctx.dispatch(rig.graphs.text.graph,  t.inputsFor("text"),  t.outputsFor("text"));
await ctx.readTensor(t.get("text", "out"));            // a completion fence
ctx.dispatch(rig.graphs.image.graph, t.inputsFor("image"), t.outputsFor("image"));
await ctx.readTensor(t.get("image", "out"), hostView); // 1 MB RGBA
```

`createEntryTensors` allocates a single MLTensor for each chain link, so the
text graph's output is the same tensor as the image graph's
`encoder_hidden_states` input, and the embedding never gets copied into JS. It
also allocates the outputs an entry marks as non-readable, since WebNN requires
every declared output to be bound at dispatch time.

The three LaTeX families decode one token at a time. You run the encoder once,
then hand the decode graph to `autoregressive()`. That function reads the
family's `contract.chaining.autoregressive` block and does the rest:

```js
ctx.dispatch(rig.graphs.encoder.graph,
             t.inputsFor("encoder"), t.outputsFor("encoder"));
await ctx.readTensor(t.get("encoder", "enc_kT0"));   // fence

const { tokens } = await autoregressive(
  ctx, rig, t, family.contract.chaining.autoregressive, { maxNewTokens: 512 },
);
```

Input names are different for each family. Texo's encoder takes `image`,
Texify's takes `pixel_values`, and IntelliTeX's takes `input_ids` and
`pad_bias`. Check the family's `contract.graphs` block for the actual names
instead of copying from here.

The full API is in [`runtime/README.md`](runtime/README.md).

### Weights, and streaming them

WebLLM entries fetch weights from the upstream repo at a pinned revision. WebNN
constants are a separate blob that isn't in git. Each entry's `manifest.json`
pins the blob by sha256 and points at where it's published:
[`ozhyhinas/webnn-catalog-sd-turbo`](https://huggingface.co/datasets/ozhyhinas/webnn-catalog-sd-turbo)
and [`ozhyhinas/webnn-catalog`](https://huggingface.co/datasets/ozhyhinas/webnn-catalog).
If you pass `null` for the constants argument the loader uses those URLs. If
you pass a base URL it fetches from there instead, so you can host a mirror.
The sha256 check runs either way.

Blobs above a certain size have a **chunk list** in the manifest. The loader
fetches one range at a time and frees it before fetching the next, so the
1.65 GiB SD-Turbo image blob turns into 32 requests with about 63 MiB in memory
at once. Chunk boundaries fall on constant boundaries, so no constant is split
across two chunks and the loader can use the fetched bytes directly without
copying. There are two chunks around 97 MiB, and those are single embedding
matrices that can't be split any further.

### Tokenizers and preprocessing

Each family includes the tokenizer files its entries were verified against.
SD-Turbo and Texo also include a small JS implementation (`ClipTokenizer` and
`TexoTokenizer`). Texify and IntelliTeX ship `tokenizer.json`, which works with
transformers.js or anything compatible. WebLLM entries use the upstream
tokenizer.

Preprocessing is your job. Each `family.json` describes exactly what the model
expects, and the verification set includes the actual bytes Chrome's canvas
produced during verification, so you can compare.

## Choosing an entry

Nothing in this repo picks an entry for you. There's no ranking function, no
fallback chain, and no "best entry" field. Which entry a given user gets is a
product decision, and it's different for every product, so it isn't encoded
here.

Start by filtering on `entry.compat.requires`, which is the set of conditions
that have to hold for the entry to work. Then order by whatever you care about:
same GPU vendor, then same browser major, then same OS major, or something else.
You'll also want to decide what happens when nothing matches, and whether an
approximating `variant` is acceptable for your users.

`entry.target` records where an entry was built and tuned. That's a performance
hint, not a requirement. Each field in it is tagged with how it was observed.
`backend.name`, `gpu.vendor`, and `browser.major` can all be read from a page.
`host.chip` can't, and a product that claims to match on it is guessing.

## How it is organized

```
families/<family-id>/family.json                    the contract
families/<family-id>/entries/<entry-id>/entry.json  one configuration
```

A **family** is one model, one task, and one I/O contract. Every entry in the
family meets that contract exactly. If a change would break any part of it,
that's a new family.

An **entry** is the family built and tuned for one configuration. Two entries in
a family compute the same thing but are spelled differently, because what's fast
on one backend usually isn't what's fast on another.

`runtimeKind` says which kind of entry you're looking at. A **WebNN** entry has a
*recipe*: the exact sequence of `MLGraphBuilder` calls that built a tuned graph,
every op in order, with the shapes the backend inferred, plus a constants blob.
A **WebLLM** entry has hash-pinned runtime JavaScript and model-library WASM,
plus a reference to an upstream model repository at a pinned revision. The two
formats are kept separate.

A recipe isn't a model format. There's no autodiff, no training metadata, and no
graph optimizer. It's the result of optimization, with the folds and rewrites
already applied.

## Reading a recipe yourself

You don't have to use the loader. A WebNN recipe is JSON, and replaying it is a
loop over the ops. There are four things the schema requires a reader to handle,
all documented in `recipe.schema.json` under `x-callForms`.

Operand names share one namespace. Graph inputs keep their own names, constants
are `k<N>`, and op results are `v<N>`. Ops appear in execution order, so every
input to an op has already been defined by the time you reach it.

Operand-valued options show up twice. An op's `inputs` array is
`[...positional operands, ...operand-valued options]`, because the options bag is
always the last argument to the builder call. To get the positional list back,
count the operand-valued options and drop that many from the end.

Some `options` keys are actually positional arguments. The recorder flattens
positional non-operand arguments into the options bag, which means the JSON by
itself can't distinguish `softmax(x, 2)` from `softmax(x, {axis: 2})`.
`x-callForms.positional` is the table that resolves this, along with `variadic`
(`concat`) and `multiOutput` (`split`).

Constants index the blob by byte offset, in increasing order. This is what lets a
chunked reader free each range once it's past it.

### Schemas

| schema | file it defines |
|---|---|
| [`recipe.schema.json`](schema/recipe.schema.json) | a graph recipe: inputs, outputs, constants, the op list |
| [`entry.schema.json`](schema/entry.schema.json) | `entry.json`: target, compat, graphs, chain, the paths |
| [`family.schema.json`](schema/family.schema.json) | `family.json`: source model, I/O contract, tokenizer |
| [`target.schema.json`](schema/target.schema.json) | `target.json`: the configuration, with `observable` |
| [`manifest.schema.json`](schema/manifest.schema.json) | `manifest.json`: constants blobs by sha256, chunk lists |
| [`artifact-manifest.schema.json`](schema/artifact-manifest.schema.json) | `artifacts.json`: hash-pinned WebLLM artifacts |
| [`measurements.schema.json`](schema/measurements.schema.json) | `measurements.json`: one row per host |
| [`catalog.schema.json`](schema/catalog.schema.json) | `catalog.json`: the index, summary rows only |

## Trying it locally

```bash
npm install
node scripts/serve.mjs
```

Then open one of the demos:

```
http://localhost:8903/demo/index.html?entry=sd-turbo-512-1step/coreml-apple-m5-pro-macos26-chrome152
http://localhost:8903/demo/qwen.html
```

The WebNN demo streams constants from the published URLs in chunks and shows
timing per stage. The Qwen demo verifies each artifact's hash before importing
it, and accepts `?entry=` to pick a Qwen family.

## Upstream models

Each family is a hand-tuned rebuild of a published model, and the upstream
license applies to the weights. Check it before you ship, especially for
commercial use.

| family | upstream | license |
|---|---|---|
| `qwen3-*-q4f16-1` | [`mlc-ai/Qwen3-*-q4f16_1-MLC`](https://huggingface.co/mlc-ai), revision-pinned | Apache-2.0 |
| `sd-turbo-512-1step` | [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo) + [`madebyollin/taesd`](https://huggingface.co/madebyollin/taesd) | see upstream |
| `texify-420` | [`vikp/texify`](https://huggingface.co/vikp/texify) | see upstream |
| `intellitex-t5-220m` | [`duanxianpi/IntelliTex`](https://huggingface.co/duanxianpi/IntelliTex) | see upstream |
| `texo-384` | Texo / FormulaNet as shipped in LatexGen | see upstream |

WebLLM entries include a `NOTICE.md` with third-party attribution.

## Contributing

If you want to add a family, build an entry for different hardware, or
contribute timing rows from your own machine, see
[`CONTRIBUTING.md`](CONTRIBUTING.md).
