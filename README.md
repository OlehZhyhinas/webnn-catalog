# webnn-catalog

Configuration-keyed, verified browser inference artifacts. The catalog carries
hand-built WebNN graph recipes and first-class WebLLM/WebGPU entries; it does
not pretend a WebLLM model library is an `MLGraphBuilder` recipe.

A **recipe** is the exact `MLGraphBuilder` call sequence that built a tuned
graph once: every op, in order, with the operand shapes the backend itself
inferred, recorded as JSON alongside a blob of constants. `runtime/loader.js`
replays it into an `MLGraph`. A WebLLM entry instead names hash-pinned runtime
JavaScript and model-library WASM plus a revision-pinned upstream model
repository. `runtimeKind` discriminates the two formats.

It is not a model format. There is no autodiff, no training metadata, no
framework, no graph optimiser. A recipe is the *output* of optimisation: the
folds, the rewrites and the spelling choices are already baked in, and the
ledger next to it says which ones, and what each was worth in milliseconds.

**The catalog stores and describes. It does not select.** Nothing here probes
your machine, ranks entries, or picks a default. That is the consuming
product's policy, and it is the one thing this repo deliberately does not have
an opinion about. See [Selection is the consumer's job](#selection-is-the-consumers-job).

## Families and entries

```
families/<family-id>/family.json                    the contract
families/<family-id>/entries/<entry-id>/entry.json  one configuration
```

A **family** is one model, one task, one I/O contract: what you hand in, what
comes back, which tokenizer turns a prompt into graph inputs, how the graphs
chain on device. Every entry of a family meets it exactly. If a change would
falsify any of it, that is a new family, not a new entry.

An **entry** is that family built and tuned for one configuration. Two entries
of a family compute the same thing; they differ in how it is spelled, because
what is fast on Core ML is not what is fast on DirectML, and what a browser
will even build changes with its version. An entry carries its own recipes,
its own constants manifest, its own verification bars, its own timings and its
own ledger.

### The configuration key

`entry.target` is the configuration the entry was **built and tuned for**. It
is produced by `scripts/probe-target.mjs` on the machine that produced the
entry, and every field is tagged with how it was observed:

| field | `observable` | what it is |
|---|---|---|
| `backend` | `browser` | Which WebNN backend the browser actually gave you, inferred from `opSupportLimits()`, plus the raw fields it was inferred from |
| `browser` | `browser` | Brand, full version, major, and the flags the harness launched with |
| `os` | `browser(partial)/host` | Name and version. A page sees a coarsened `platformVersion`; the build id needs a shell |
| `gpu` | `browser` | WebGPU `adapter.info` vendor / architecture / device / description, plus a limits summary |
| `host` | `host` | Chip name, GPU core count, RAM. Never visible to a page |
| `perfClass` | `browser(probe)` | A measured f16 matmul throughput, as a lower bound, so two machines can be put in rough classes |

The split matters because it tells a consumer what it can actually match on at
runtime. `backend.name`, `gpu.vendor` and `browser.major` are readable in the
page. `host.chip` is not, and no product should pretend otherwise.

`entry.compat` states **facts, not policy**: `requires` is what must hold for
the graphs to build and compute correctly (for the entry below, only that the
backend is Core ML), and `ops` is the set of `MLGraphBuilder` methods the
recipes call, so a browser missing one cannot build them whatever else matches.
Everything else in `target` is where the entry was tuned and timed, which is a
weaker claim, and `measurements.json` says exactly which machines the numbers
came from.

The **entry id** is `<backend>-<chip slug>-<os><major>-<browser><major>`. It is
a readable summary of `target` for humans reading a directory listing. Nothing
parses it.

## The catalog today

| family | entry | variant | new prompt | quality | measured on |
|---|---|---|---:|---|---|
| [`sd-turbo-512-1step`](families/sd-turbo-512-1step/) | [`coreml-apple-m5-pro-macos26-chrome152`](families/sd-turbo-512-1step/entries/coreml-apple-m5-pro-macos26-chrome152/) | `exact` | **68.4 ms** | PSNR 44.28 dB vs the fp32 chain | WebNN / Core ML, Apple M5 Pro, macOS 26.6.2, Chrome 152 |
| [`texo-384`](families/texo-384/) | [`coreml-apple-m5-pro-macos26-chrome152`](families/texo-384/entries/coreml-apple-m5-pro-macos26-chrome152/) | `exact` | **22.7 ms** per image | greedy tokens identical to fp32 on 18/18 benchmark images | WebNN / Core ML, Apple M5 Pro, macOS 26.6.2, Chrome 152 |
| [`intellitex-t5-220m`](families/intellitex-t5-220m/) | [`coreml-apple-m5-pro-macos26-chrome152`](families/intellitex-t5-220m/entries/coreml-apple-m5-pro-macos26-chrome152/) | `exact` | **146.2 ms** | tokens identical to fp32 Hugging Face on 15/15 items | WebNN / Core ML, Apple M5 Pro, macOS 26.6.2, Chrome 152 |

For context, on the same machine and the same model: ONNX Runtime Web on WebGPU
is 916 ms, the same demo on ORT's WebNN EP is 237 ms, and native PyTorch MPS is
94.5 ms. For Texo, LatexGen's own ONNX Runtime Web path takes 770 ms per image
(WASM fp32) or 780 ms (WebGPU); "new prompt" for that family means one
preprocessed image in, a LaTeX token sequence out, and the entry is
autoregressive: a decode graph that runs 16 greedy steps per dispatch over
static caches, driven by `runtime/loader.js`'s `autoregressive()` from the
family's `contract.chaining.autoregressive` block.

Machine-readable index: [`catalog.json`](catalog.json), which holds summary rows
only. The entry directory is authoritative for everything in them.

### Qwen3-0.6B WebLLM/WebGPU

[`qwen3-0.6b-q4f16-1`](families/qwen3-0.6b-q4f16-1/) is the first
`runtimeKind: "webllm"` family. Its Apple M5 Pro entry uses subgroup-32,
chunk-256 GPU argmax, GEMV `TR=32`, and K=4 GPU-resident greedy decode.
On six warm-model, fresh-prompt interleaved rounds under machine load it
measured **250.655 tokens/s**, **1.692x** the same artifact at K=1. The
approximately 3.4 ms/token quiet result is recorded only as a projection.

A second entry, `…-sg32-burst4-flush64`, keeps that library and burst and
adds three runtime flags (single compute pass, `queue.submit` every 64
dispatches so the GPU executes while JS still encodes, bind-group reuse).
Paired live against the first entry in the same rotation it measured
**312.6 tokens/s**, **1.243x** (IQR 1.229–1.260), byte-identical output.
The ratio was taken under CPU load and is an upper bound for a quiet machine;
no quiet figure is recorded. The demo defaults to this entry.

The [Qwen demo](demo/qwen.html) fetches the runtime bundle and model WASM,
verifies their byte counts and SHA-256 hashes, imports the verified runtime,
then fetches model files from an immutable upstream Hugging Face revision.
The fast path is limited to exact greedy requests; sampling, logprobs,
penalties, grammar/structured output, logit bias, and custom logit processors
automatically retain the ordinary one-step WebLLM path.

### Qwen3-1.7B WebLLM/WebGPU

[`qwen3-1.7b-q4f16-1`](families/qwen3-1.7b-q4f16-1/) uses a
revision-pinned upstream model with a subgroup-32, chunk-256, GEMV-`TR=32`
model library. Its M5 Pro entry adds K=4 exact-greedy decode and submits the
single WebGPU compute pass every 32 dispatches.

Across six warm-model, fresh-prompt paired rounds under ORCA it measured
**139.86 tokens/s** (**7.15 ms/token**), **1.371x** the published subgroup-32
path. All six fresh outputs and four quality outputs were byte-identical.
The 6.5 ms/token quiet figure is an upper-bound projection, not a measurement.

The Qwen demo accepts an optional `entry` query parameter, so either Qwen
family can be loaded through the same verified UI.

**Pending a dump:** the ToDo (Token Downsampling) fast variant of the SD-Turbo
family runs at 54.6 ms, about 20% faster, for a visibly different image (PSNR
24.3 dB against the exact pipeline). The workbench measured it end to end but
never recorded its `MLGraphBuilder` call sequence, so there is no recipe and
therefore no `todo2-nearest` entry. The numbers are in the entry's
[`provenance.md`](families/sd-turbo-512-1step/entries/coreml-apple-m5-pro-macos26-chrome152/provenance.md#the-optional-fast-variant);
the entry appears when a dump does.

A second entry, `…-sg32-burst1-flush32-lookahead1`, keeps that model
library and the submit cadence, and changes the decode loop: one decode step
stays queued on the GPU while the current burst is read back, so the GPU
never idles at a burst boundary, and the burst shrinks to one token, which
streams every token and computes nothing past EOS inside the stream. Paired
live against the first entry in the same rotation over twelve fresh-prompt
rounds under ORCA it measured **158.66 tokens/s**, **1.136x** (IQR
1.065–1.189), byte-identical output; the one speculative step past EOS adds
about 4 ms to the TTFT of a back-to-back request (1.068x end-to-end). The
ratio was taken under CPU load and is an upper bound for a quiet machine.
The loader applies the new `runtime.config.lookahead` flag; a loader that
predates it runs plain K=1 on the same bundle.

### Qwen3-4B WebLLM/WebGPU

[`qwen3-4b-q4f16-1`](families/qwen3-4b-q4f16-1/) references the immutable
upstream 4B weights and adds a model-specific subgroup-32, chunk-256,
GEMV-`TR=32` library. Its M5 Pro entry uses K=4 exact-greedy decode, one
compute pass, and a submit every 32 dispatches. The unchanged WebLLM runtime
artifact is reused rather than duplicated.

Across six warm-model, fresh-prompt paired rounds under ORCA it measured
**73.85 tokens/s** (**13.54 ms/token**), **1.285x** the fully published
WebLLM 0.2.84 subgroup-32 path. All six fresh outputs and completion-token
counts, plus all four quality outputs, were byte-identical. There is no
historical quiet 4B number, so no quiet projection is claimed.

The Qwen demo accepts `?entry=` and continues to support every Qwen family;
the existing 0.6B default is unchanged.

## Adding an entry

Three commands, in this order.

```bash
# 1. Describe the machine you are about to build on.
node scripts/probe-target.mjs --out target.json

# 2. Write the entry from what the tuning run produced.
node scripts/add-entry.mjs \
  --family sd-turbo-512-1step \
  --target target.json \
  --recipe image=<dump>.json --recipe text=<dump>.json \
  --constants image=<blob>.bin --constants text=<blob>.bin \
  --chain text.out=image.encoder_hidden_states \
  --variant exact \
  --verification <dir> --provenance <ledger>.md \
  --measurements <results>.json \
  --produced-by "webnn-workbench@<commit>"

# 3. Check everything, against the schemas and against the hardware.
node scripts/validate.mjs
node scripts/verify.mjs --entry sd-turbo-512-1step/<entry-id>
```

`add-entry.mjs` reads the recipes to fill in the entry's I/O, hashes the
constants blobs **where they lie** (they are 1.65 GB and 649 MB and they never
enter git), writes `entry.json`, `manifest.json` and `measurements.json`, copies
the verification set and the ledger in, adds a summary row to `catalog.json`,
and refuses to overwrite an existing entry without `--force`. It validates what
it wrote before it exits.

The family itself is authored by hand: `family.json` is the contract, and a tool
that guessed at a contract would be guessing at the one thing entries are not
allowed to disagree about.

A second machine running the same entry is a **row**, not an entry:

```bash
node scripts/probe-target.mjs --out target.json
node scripts/add-measurement.mjs --entry <family>/<entry-id> \
  --results <results>.json --target target.json --protocol e2e
```

Each row carries its own host fingerprint, so one entry can accumulate timings
from several machines without becoming several entries.

## Loading an entry

`runtime/loader.js` is the serving interface: hand it an entry, get finished
graphs.

```js
import {
  loadEntry, createEntryTensors, assertCoreMLFingerprint,
} from "./runtime/loader.js";

const dir   = "/families/sd-turbo-512-1step/entries/coreml-apple-m5-pro-macos26-chrome152";
const entry = await (await fetch(`${dir}/entry.json`)).json();

const ctx = await navigator.ml.createContext({ deviceType: "gpu" });
assertCoreMLFingerprint(ctx);   // see below; this line saves afternoons

const rig = await loadEntry(entry, "/weights", ctx, { baseUrl: dir });
// -> { graphs: {text, image}, chain, inputs, outputs, manifest, stats }

const t = await createEntryTensors(ctx, rig);
ctx.writeTensor(t.get("text", "input_ids"), ids);

ctx.dispatch(rig.graphs.text.graph,  t.inputsFor("text"),  t.outputsFor("text"));
await ctx.readTensor(t.get("text", "out"));            // a completion fence
ctx.dispatch(rig.graphs.image.graph, t.inputsFor("image"), t.outputsFor("image"));
await ctx.readTensor(t.get("image", "out"), hostView); // 1 MB, into your buffer
```

`createEntryTensors` reads `entry.graphs.chain` and allocates **one** MLTensor
for each link, so the text graph's output *is* the image graph's
`encoder_hidden_states` and the embedding never crosses into JS. It also
allocates the outputs an entry marks non-readable, because WebNN requires every
declared graph output to be bound at dispatch, debug outputs included.

Constants arrive either as one fetch or, if the manifest lists `chunks`, as
ranged fetches released as they are consumed, so a 1.65 GB blob does not have to
sit in the JS heap in one piece. Both paths are exercised by `scripts/verify.mjs`
and produce the same output byte for byte.

Full API: [`runtime/README.md`](runtime/README.md).

### `assertCoreMLFingerprint` is not optional

Chromium gates the Core ML backend on a **non-incognito** profile. An
off-the-record profile, which is what `chromium.launch()` and an incognito
window both give you, silently falls back to TFLite/XNNPACK on the CPU. Nothing
errors. The graph builds in milliseconds instead of ~25 seconds and runs about
50x slower.

| | Core ML | TFLite/XNNPACK |
|---|---|---|
| `opSupportLimits().preferredInputLayout` | `nchw` | `nhwc` |
| `input.rankRange.max` | 5 | 8 |

Call it right after `createContext()`. It is the same fingerprint
`probe-target.mjs` writes into `target.backend`.

## The data contract

The schemas in [`schema/`](schema/) are the interface, and `validate.mjs`
enforces them over the whole repo.

| schema | file it defines |
|---|---|
| [`recipe.schema.json`](schema/recipe.schema.json) | a graph recipe: inputs, outputs, constants, the op list |
| [`entry.schema.json`](schema/entry.schema.json) | `entry.json`: target, compat, graphs, chain, the paths |
| [`family.schema.json`](schema/family.schema.json) | `family.json`: source model, I/O contract, tokenizer, scheduler |
| [`target.schema.json`](schema/target.schema.json) | `target.json`: the configuration, field by field, with `observable` |
| [`manifest.schema.json`](schema/manifest.schema.json) | `manifest.json`: constants blobs by sha256, recipe hashes |
| [`measurements.schema.json`](schema/measurements.schema.json) | `measurements.json`: one row per host, each with its own fingerprint |
| [`catalog.schema.json`](schema/catalog.schema.json) | `catalog.json`: the index, summary rows only |

### The recipe IR

A product does not have to use `runtime/loader.js`. The recipe is JSON, and
replaying it is a loop:

```jsonc
{
  "version": 1,
  "label": "sd-turbo-512-1step/image",
  "layout": "nhwc",
  "inputs":  [{ "name": "sample", "dataType": "float16", "shape": [1,64,64,4] }],
  "outputs": [{ "name": "out", "operand": "v1488", "dataType": "int32", "shape": [1,512,512] }],
  "constants": {
    "k0": { "dataType": "float16", "shape": [320,3,3,4], "byteOffset": 0, "byteLength": 23040, "tag": "conv_in" }
  },
  "ops": [
    { "id": 1, "type": "conv2d", "inputs": ["sample","k0","k1"], "output": "v1",
      "outputShape": [1,64,64,320], "outputDataType": "float16",
      "options": { "strides":[1,1], "padding":[1,1,1,1], "dilations":[1,1],
                   "inputLayout":"nhwc", "filterLayout":"ohwi", "bias":"k1" },
      "tag": "conv_in" }
  ]
}
```

Four things the schema says that a reader has to act on. All four are recorded
in `recipe.schema.json` under `x-callForms`, so they are data, not folklore.

**One operand namespace.** Graph inputs keep their own names, constants are
`k<N>`, op results are `v<N>`. Ops are in order, so an op's inputs are always
already defined.

**Operand-valued options appear twice.** An op's `inputs` is
`[...positional operands, ...operand-valued options]`, in that order, because an
options bag is always the last argument. `conv2d`'s bias is both `options.bias`
and the tail of `inputs`. Shear the operand-valued options off the tail by count
to recover the positional list.

**Some `options` keys are really positional arguments.** The recorder flattens a
call's positional non-operand arguments into the same bag as the real options
dictionary, so the JSON alone cannot tell `softmax(x, 2)` from
`softmax(x, {axis: 2})`. `x-callForms.positional` is the recorder's table,
inverted:

```js
reshape: ["newShape"],  expand: ["newShape"],  softmax: ["axis"],
cast: ["type"],         concat: ["axis"],      split: ["splits"],
tile: ["repetitions"],  pad: ["beginningPadding", "endingPadding"],
argMin: ["axis"],       argMax: ["axis"],
```

Plus `variadic` (`concat` takes an array of operands, not one) and `multiOutput`
(`split` returns several, and its op carries `outputs` / `outputShapes` arrays
alongside `output` / `outputShape`).

**Constants index the blob by byte offset, in increasing order.** That is what
lets a chunked reader release each range as it goes. A constant that straddles a
chunk seam is copied out and stitched.

A version 2 of the recipe schema should record the argument list positionally,
or name the call form explicitly, and then the positional table can go.

## Selection is the consumer's job

Nothing in this repo matches an entry to a machine. There is no `query.js`, no
ranking function, no fallback chain, and no "best entry" field. The catalog
publishes facts; deciding which entry a given user gets is product policy, it
changes per product, and it is exactly the kind of thing that rots when it is
frozen into a data repo.

What a product might do, as **guidance only**, not part of the catalog:

```js
// 1. Probe, in the page, using only what is browser-observable.
const ctx     = await navigator.ml.createContext({ deviceType: "gpu" });
const limits  = ctx.opSupportLimits();
const backend = limits.preferredInputLayout === "nchw" && limits.input.rankRange.max === 5
  ? "coreml" : limits.preferredInputLayout === "nhwc" ? "tflite" : "unknown";
const adapter = await navigator.gpu.requestAdapter();
const ua      = await navigator.userAgentData.getHighEntropyValues(["platformVersion"]);

// 2. Filter on what MUST hold: entry.compat.requires, then the ops the browser
//    actually exposes.
const family  = catalog.families["sd-turbo-512-1step"];
let usable    = family.entries.filter((e) => e.backend === backend);

// 3. Prefer, by your own policy, not the catalog's. For example: the same GPU
//    vendor, then the same browser major, then the same OS major.
usable.sort(byYourPreference(adapter.info.vendor, browserMajor, osMajor));

// 4. Fall back to any entry of the family with the same backend, and accept
//    that its timings were measured somewhere else.
const chosen = usable[0] ?? family.entries.find((e) => e.backend === backend);
```

Two things worth deciding deliberately when you write that: whether an
approximating `variant` is acceptable to your users at all, and what you do when
nothing matches. The catalog will not decide either for you.

## Running it

```bash
npm install

# Validate every file against schema/, plus the cross-file facts.
node scripts/validate.mjs

# Verify an entry on real hardware, against a local weights directory
# (default: the workbench IR dir).
node scripts/verify.mjs --entry sd-turbo-512-1step/coreml-apple-m5-pro-macos26-chrome152 \
  --weights ../webnn-workbench/bench/webnn/ir

# ...or serve the catalog and open the demo.
node scripts/serve.mjs --weights ../webnn-workbench/bench/webnn/ir
open "http://localhost:8903/demo/index.html?entry=sd-turbo-512-1step/coreml-apple-m5-pro-macos26-chrome152"
```

`scripts/verify.mjs` launches Chrome with a **persistent** profile and the WebNN
flags, builds every graph the entry declares, runs the family's reference case,
checks the sha256 of the RGBA readback and the PSNR against two reference
images, times 20 runs and prints a table. `--chunk-mb N` exercises the ranged
constant path. It runs the entry you name and reports what happened; it does not
look for an entry that suits the machine.

`scripts/validate.mjs` is dependency-free. It implements the subset of JSON
Schema the schemas use, refuses a schema keyword it does not know rather than
passing it silently, and then checks what a schema cannot: that referenced files
exist, that each recipe still hashes to what its manifest says, that an entry's
declared I/O equals its recipes' and meets its family's contract, that every
chain link typechecks, and that the index agrees with the entries it indexes.

## What is in the repo, and what is not

Recipes, manifests, verification references, ledgers, the schemas, the loader
and the tooling are here. **The constants blobs are not**: 1.65 GB and 649 MB do
not belong in git. They are the published artifact, pinned by sha256 in each
entry's manifest and fetched from a URL.

```
catalog.json                    the index: families -> entries, summary rows only
schema/                         JSON Schema for every file type in here
families/<family>/
  family.json                   the contract: source model, I/O, tokenizer, chaining
  tokenizer.js, tokenizer/      shared by every entry of the family
  entries/<entry>/
    entry.json                  the configuration record
    recipe.image.json           1441 ops, 655 constants
    recipe.text.json            603 ops, 281 constants
    manifest.json               blob sizes, sha256s, URLs; recipe hashes
    measurements.json           one row per host, each with its own fingerprint
    verification/               the reference case, this entry's bars, references
    provenance.md               the ledger
runtime/
  loader.js                     recipe -> MLGraph, entry -> graphs; no dependencies
  README.md                     API, constant sources, the chaining pattern
scripts/
  probe-target.mjs              write target.json for this machine
  add-entry.mjs                 create an entry from a tuning run's output
  add-measurement.mjs           append a row from another host
  validate.mjs                  schema + cross-file checks over the whole catalog
  verify.mjs                    Playwright, real Chrome, real Core ML
  serve.mjs                     static server with Range support, for the demo
  publish-weights.sh            upload the blobs, write their URLs into a manifest
demo/index.html                 prompt box, image, per-stage ms; ?entry=<family>/<id>
```

## Five conventions

**A recipe is the exact call sequence, with backend-inferred shapes.** Not a
re-derivation. If the recorder computes the shapes itself, the recipe can
disagree with the graph that was measured, and eventually will. `manifest.json`
pins each recipe's sha256 so a hand edit is detectable, and `validate.mjs`
checks it.

**Constants never enter git.** A separate blob, pinned by sha256, whose byte
offsets the recipe indexes.

**A number without a machine, an OS, a browser build and a protocol is not a
measurement.** Every row in `measurements.json` carries all four, including how
the A/B was run and what the noise floor is. A row's `loadAvg` is never borrowed
from somewhere else.

**Every entry has a ledger.** What was folded and what it was worth; what was
tried and rejected, with the number that killed it; and what the backend turned
out to be like. The op list survives on its own. The reasoning does not.

**The catalog describes; it does not choose.** An entry says what it is and
where it was built. It never says it is the right one.
