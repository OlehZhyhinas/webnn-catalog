# webnn-catalog

Pre-tuned browser inference, published as data. Point a loader at an entry and
get a running model. You don't need a build step, a server, or a native install.

SD-Turbo generates a 512×512 image from a new prompt in **68 ms** on an M5 Pro
in Chrome. ONNX Runtime Web on WebGPU takes 916 ms on the same machine for the
same model; native PyTorch on MPS takes 94.5 ms.

That gap is the point. This is not a new runtime. The WebNN and WebGPU your
browser already ships are driven in an order somebody tuned by hand and then
recorded so you don't have to.

There is no npm package. `runtime/` holds two dependency-free ES modules.
Vendor them, or serve them from your own static host.

---

## What you can run

### Image and LaTeX: WebNN / Core ML

| family | task | in → out | download | new prompt |
|---|---|---|---:|---:|
| [`sd-turbo-512-1step`](families/sd-turbo-512-1step/) | text → image | prompt → 512×512 RGBA | 2.41 GB | **68.4 ms** |
| [`texo-384`](families/texo-384/) | image → LaTeX | 384×384 grayscale crop → LaTeX | 44 MB | **25.9 ms** |
| [`texify-420`](families/texify-420/) | image → LaTeX | 420×420 RGB crop → LaTeX | 627 MB | **145.7 ms** |
| [`intellitex-t5-220m`](families/intellitex-t5-220m/) | text → LaTeX | English → LaTeX | 1.04 GB | **146.2 ms** |

Fidelity against the fp32 reference each was built from: SD-Turbo PSNR 44.28 dB;
Texo and Texify greedy tokens identical on 18/18 benchmark images; IntelliTeX
tokens identical to fp32 Hugging Face on 15/15 items.

For Texo, LatexGen's own ONNX Runtime Web path takes 770 ms per image on WASM
fp32 or 780 ms on WebGPU, against this catalog's 25.9 ms.

### Chat: WebLLM / WebGPU

OpenAI-compatible chat, exact greedy decode. Every family ships two entries
except the 8B; they share a model library and differ in decode tuning.

| family | entry | download | throughput | vs baseline |
|---|---|---:|---:|---:|
| [`qwen3-0.6b-q4f16-1`](families/qwen3-0.6b-q4f16-1/) | `…-sg32-burst4` | 359 MB | 250.7 tok/s | 1.692× |
| | `…-sg32-burst4-flush64` | 359 MB | **312.6 tok/s** | 1.243× |
| [`qwen3-1.7b-q4f16-1`](families/qwen3-1.7b-q4f16-1/) | `…-sg32-burst4-flush32` | 996 MB | 139.9 tok/s | 1.371× |
| | `…-burst1-flush32-lookahead1` | 996 MB | **158.7 tok/s** | 1.136× |
| [`qwen3-4b-q4f16-1`](families/qwen3-4b-q4f16-1/) | `…-sg32-burst4-flush32` | 2.29 GB | 73.9 tok/s | 1.285× |
| | `…-burst1-flush32-lookahead1` | 2.29 GB | **78.5 tok/s** | 1.063× |
| [`qwen3-8b-q4f16-1`](families/qwen3-8b-q4f16-1/) | `…-sg32-burst4-flush32` | 4.64 GB | **46.2 tok/s** | 1.187× |

Output is byte-identical to the baseline each entry was paired against. The
`lookahead1` entries stream every token; the `burst4` entries emit in blocks of
four.

Every number on this page was measured on Apple M5 Pro / macOS 26.6.2 /
Chrome 152. Each entry directory carries its own `measurements.json` and
`provenance.md` with the baseline, the protocol, the host and the conditions.
Read those before quoting a figure anywhere it matters.

Machine-readable index: [`catalog.json`](catalog.json).

## What you need

- **Chrome 152 or newer.**
- **WebNN entries:** WebNN enabled, and a **non-incognito profile**. Chromium
  gates the Core ML backend on it. An off-the-record profile silently falls
  back to CPU and runs about 50× slower with no error. Call
  `assertCoreMLFingerprint(ctx)` and fail loudly instead.
- **WebLLM entries:** WebGPU with `shader-f16` and subgroup size 32. The loader
  checks before it downloads anything.
- **Room for the download.** See the tables. WebNN constants stream in chunks,
  so peak resident memory stays ~64–98 MiB regardless of blob size.

Every entry here was tuned on Apple silicon. `entry.compat.requires` states what
must hold for an entry to work at all. Filter on that, and see
[choosing an entry](#choosing-an-entry).

## Using it

Two loaders, selected by an entry's `runtimeKind`.

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

`loadWebLLMFromUrl()` checks subgroup-32 support, fetches the entry's artifact
manifest, verifies the byte count and SHA-256 of both the runtime JavaScript and
the model-library WASM, and only then exposes them through temporary blob URLs.
Model weights come from a revision-pinned upstream Hugging Face repository. The
catalog never rehosts them. `dispose()` unloads the engine and revokes the URLs.

The exact-greedy fast path applies when `temperature` is 0 and no logprobs,
penalties, logit bias, grammar, structured output, or custom logit processor is
active. Anything else transparently uses WebLLM's ordinary sampler with the
same API.
`isGreedyBurstEligible()` exposes that rule if you want to check first.

### Image and LaTeX: `runtimeKind: "webnn"`

```js
import {
  loadEntry, createEntryTensors, assertCoreMLFingerprint,
} from "./runtime/loader.js";

const entry = await (await fetch(`${dir}/entry.json`)).json();

const ctx = await navigator.ml.createContext({ deviceType: "gpu" });
assertCoreMLFingerprint(ctx);   // this line saves afternoons

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

`createEntryTensors` allocates **one** MLTensor per chain link, so the text
graph's output *is* the image graph's `encoder_hidden_states` and the embedding
never crosses into JS. It also allocates outputs an entry marks non-readable,
because WebNN requires every declared output to be bound at dispatch.

The three LaTeX families decode token by token. Run the encoder once, then hand
the decode graph to `autoregressive()`, which is driven entirely by the family's
`contract.chaining.autoregressive` block:

```js
ctx.dispatch(rig.graphs.encoder.graph,
             t.inputsFor("encoder"), t.outputsFor("encoder"));
await ctx.readTensor(t.get("encoder", "enc_kT0"));   // fence

const { tokens } = await autoregressive(
  ctx, rig, t, family.contract.chaining.autoregressive, { maxNewTokens: 512 },
);
```

Input names differ per family. Texo's encoder takes `image`, Texify's takes
`pixel_values`, IntelliTeX's takes `input_ids` plus `pad_bias`. Read the
family's `contract.graphs` block rather than copying names.

Full API: [`runtime/README.md`](runtime/README.md).

### Weights, and streaming them

WebLLM entries pull weights from a revision-pinned upstream repo. WebNN
constants are a separate blob, never in git, pinned by sha256 in each entry's
`manifest.json` and published to
[`ozhyhinas/webnn-catalog-sd-turbo`](https://huggingface.co/datasets/ozhyhinas/webnn-catalog-sd-turbo)
and [`ozhyhinas/webnn-catalog`](https://huggingface.co/datasets/ozhyhinas/webnn-catalog).
Pass `null` and the loader uses those URLs; pass a base URL to serve your own
mirror, and the manifest's sha256 still pins what you get.

Any blob big enough to need one carries a **chunk list**, so you never hold a
whole blob. The loader fetches one range at a time and releases it before the next: the 1.65 GiB
SD-Turbo image blob becomes 32 requests with 63 MiB resident. Boundaries sit on
constant starts, so nothing straddles a seam and every view stays zero-copy. The
two ~97 MiB peaks are single embedding matrices, which no chunking can split.

### Tokenizers and preprocessing

Each family ships the tokenizer files its entries were verified against.
SD-Turbo and Texo include a small JS implementation (`ClipTokenizer`,
`TexoTokenizer`); Texify and IntelliTeX ship `tokenizer.json` for transformers.js
or equivalent. WebLLM entries get their tokenizer from upstream.

Preprocessing stays on your side, and each `family.json` states the exact recipe.
The verification set holds what Chrome's canvas actually produced, byte for byte.

## Choosing an entry

**Nothing here picks an entry for you.** There is no ranking function, no
fallback chain, and no "best entry" field. The catalog publishes facts. Which
entry a given user gets is product policy. It changes per product, and it rots
when frozen into a data repo.

Filter on `entry.compat.requires`: what must hold for the entry to work at all.
Then prefer by your own policy: same GPU vendor, then browser major, then OS
major. Decide deliberately what you do when nothing matches, and whether an
approximating `variant` is acceptable to your users.

`entry.target` is where an entry was built and tuned. It is a hint about
performance, not a requirement. Every field is tagged with how it was
observed. `backend.name`, `gpu.vendor` and `browser.major` are readable in the
page; `host.chip` is not, and no product should pretend otherwise.

## How it is organized

```
families/<family-id>/family.json                    the contract
families/<family-id>/entries/<entry-id>/entry.json  one configuration
```

A **family** is one model, one task, and one I/O contract. Every entry of a family
meets it exactly; if a change would falsify any of it, that is a new family.

An **entry** is that family built and tuned for one configuration. Two entries of
a family compute the same thing and differ in how it is spelled, because what is
fast on one backend is not what is fast on another.

`runtimeKind` says which kind of entry it is. A **WebNN** entry carries a
*recipe*: the exact `MLGraphBuilder` call sequence that built a tuned graph once,
every op in order, with the shapes the backend inferred, alongside a constants
blob. A **WebLLM** entry instead names hash-pinned runtime JavaScript and
model-library WASM plus a revision-pinned upstream model repository. The catalog
does not pretend a model library is a recipe.

A recipe is not a model format. It has no autodiff, no training metadata, and no
graph optimizer. It is the *output* of optimization, with the folds and rewrites
already baked in.

## Reading a recipe yourself

You don't have to use the loader. A WebNN recipe is JSON and replaying it is a
loop. Four things the schema says that a reader must act on, all recorded in
`recipe.schema.json` under `x-callForms`:

**One operand namespace.** Inputs keep their names, constants are `k<N>`, results
are `v<N>`. Ops are in order, so an op's inputs are always already defined.

**Operand-valued options appear twice.** An op's `inputs` is
`[...positional operands, ...operand-valued options]`, because an options bag is
always the last argument. Shear them off the tail by count to recover the
positional list.

**Some `options` keys are really positional arguments.** The recorder flattens
positional non-operand arguments into the options bag, so the JSON alone cannot
tell `softmax(x, 2)` from `softmax(x, {axis: 2})`. `x-callForms.positional` is
that table, inverted, plus `variadic` (`concat`) and `multiOutput` (`split`).

**Constants index the blob by byte offset, in increasing order.** That is what
lets a chunked reader release each range as it goes.

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

Then open either demo:

```
http://localhost:8903/demo/index.html?entry=sd-turbo-512-1step/coreml-apple-m5-pro-macos26-chrome152
http://localhost:8903/demo/qwen.html
```

The WebNN demo streams constants from the published URLs chunk by chunk and
shows per-stage milliseconds. The Qwen demo verifies every artifact hash before
importing it, and takes `?entry=` to load any Qwen family.

## Upstream models

Each family is a hand-tuned rebuild of a published model. The upstream license
governs its weights. Check it before shipping, particularly commercially.

| family | upstream | license |
|---|---|---|
| `qwen3-*-q4f16-1` | [`mlc-ai/Qwen3-*-q4f16_1-MLC`](https://huggingface.co/mlc-ai), revision-pinned | Apache-2.0 |
| `sd-turbo-512-1step` | [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo) + [`madebyollin/taesd`](https://huggingface.co/madebyollin/taesd) | see upstream |
| `texify-420` | [`vikp/texify`](https://huggingface.co/vikp/texify) | see upstream |
| `intellitex-t5-220m` | [`duanxianpi/IntelliTex`](https://huggingface.co/duanxianpi/IntelliTex) | see upstream |
| `texo-384` | Texo / FormulaNet as shipped in LatexGen | see upstream |

WebLLM entries carry a `NOTICE.md` with full third-party attribution.

## Conventions

**The catalog describes; it does not choose.** An entry says what it is and where
it was built. It never says it is the right one.

**A recipe is the exact call sequence, with backend-inferred shapes.** Not a
re-derivation. `manifest.json` pins each recipe's sha256.

**Constants never enter git.** A separate blob, pinned by sha256, whose byte
offsets the recipe indexes.

**A number without a machine, an OS, a browser build and a protocol is not a
measurement.** Every row in `measurements.json` carries all four.

**Every entry has a ledger.** `provenance.md` records what was folded and what it
was worth, and what was tried and rejected with the number that killed it.

## Contributing

Adding a family, building an entry for new hardware, or contributing a timing row
from your machine: see [`CONTRIBUTING.md`](CONTRIBUTING.md).
