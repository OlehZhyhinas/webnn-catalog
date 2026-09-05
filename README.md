# webnn-catalog

Hand-built WebNN graphs, stored as data.

A **recipe** is the exact `MLGraphBuilder` call sequence that built a tuned
graph once: every op, in order, with the operand shapes the backend itself
inferred, recorded as JSON alongside a blob of constants. `runtime/loader.js`
replays it into an `MLGraph`. That is the whole idea.

It is not a model format. There is no autodiff, no training metadata, no
framework, no graph optimiser. A recipe is the *output* of optimisation: the
folds, the rewrites and the spelling choices are already baked in, and the
README next to it says which ones, and what each was worth in milliseconds.

## Why store a graph this way

A tuned WebNN graph is usually a few thousand lines of emitter code with the
tuning decisions scattered through it as flags. Nobody can read what the graph
*is* from that, the emitter and the graph can silently disagree, and the graph
cannot cross a repo boundary.

As a recipe, the graph is inspectable, diffable, and separable from the code
that generated it. It also replays about 350x faster than it emits: 10 ms for both
graphs' 2044 ops, against the 3.57 s the emitter that produced them spent.

## Catalog

| entry | what | new prompt | quality | backend |
|---|---|---:|---|---|
| [`sd-turbo-512-1step`](models/sd-turbo-512-1step/) | SD-Turbo 512x512, one Euler step. Two graphs: OpenCLIP text encoder (603 ops), and UNet + Euler + TAESD + RGBA pack (1441 ops). | **67.85 ms** | PSNR 44.28 dB vs the fp32 chain | WebNN / Core ML, Apple M5 Pro |

For context, on the same machine and the same model: ONNX Runtime Web on WebGPU
is 916 ms, the same demo on ORT's WebNN EP is 237 ms, and native PyTorch MPS is
94.5 ms.

Machine-readable index: [`catalog.json`](catalog.json).

## Loading a recipe

```js
import {
  loadRecipe, constantSource, assertCoreMLFingerprint,
} from "./runtime/loader.js";

const ctx = await navigator.ml.createContext({ deviceType: "gpu" });
assertCoreMLFingerprint(ctx);   // see below; this line saves afternoons

const manifest = await (await fetch("models/sd-turbo-512-1step/manifest.json")).json();
const recipe   = await (await fetch("models/sd-turbo-512-1step/recipe.image.json")).json();

const image = await loadRecipe(
  recipe,
  constantSource(manifest.constants.image, { baseUrl: "/weights" }),
  ctx,
);

// image.inputs / image.outputs are {name: {dataType, shape}}. Bind all of them.
ctx.dispatch(image.graph, ins, outs);
```

Constants arrive either as one fetch or, if the manifest lists `chunks`, as
ranged fetches that are released as they are consumed, so a 1.65 GB blob does
not have to sit in the JS heap in one piece. Both paths are exercised by
`scripts/verify.mjs` and produce the same output byte for byte.

Full API, the recipe schema and the MLTensor chaining pattern:
[`runtime/README.md`](runtime/README.md).

### `assertCoreMLFingerprint` is not optional

Chromium gates the Core ML backend on a **non-incognito** profile. An
off-the-record profile, which is what `chromium.launch()` and an incognito
window both give you, silently
falls back to TFLite/XNNPACK on the CPU. Nothing errors. The graph builds in
milliseconds instead of ~25 seconds and runs about 50x slower.

| | Core ML | TFLite/XNNPACK |
|---|---|---|
| `opSupportLimits().preferredInputLayout` | `nchw` | `nhwc` |
| `rankRange.max` | 5 | 8 |

Call it right after `createContext()`.

## Running it

```bash
npm install

# Verify against a local weights directory (defaults to the workbench IR dir)
node scripts/verify.mjs --weights ../webnn-workbench/bench/webnn/ir

# ...or serve the catalog and open the demo
node scripts/serve.mjs --weights ../webnn-workbench/bench/webnn/ir
open http://localhost:8903/demo/index.html
```

`scripts/verify.mjs` launches Chrome with a **persistent** profile and the WebNN
flags, replays both recipes, runs the reference prompt, checks the sha256 of the
RGBA readback and the PSNR against two reference images, times 20 runs and
prints a table. `--chunk-mb N` exercises the ranged constant path.

The demo is a prompt box, a canvas and per-stage timings. It needs a weights
directory served alongside it; once `manifest.constants[*].url` is filled in by
`scripts/publish-weights.sh`, it needs nothing local at all.

## What is in the repo, and what is not

Recipes, manifests, verification references and the loader are here. **The
constants blobs are not**: 1.65 GB and 649 MB do not belong in git. They are the
published artifact, pinned by sha256 in each manifest and fetched from a URL.
`scripts/publish-weights.sh` uploads them and writes the URLs back into the
manifest; it does not run on its own.

```
README.md              this file
catalog.json           machine-readable index
runtime/
  loader.js            replay a recipe into an MLGraph; no dependencies
  README.md            API, recipe schema, MLTensor chaining
models/sd-turbo-512-1step/
  recipe.image.json    1441 ops, 655 constants
  recipe.text.json     603 ops, 281 constants
  manifest.json        blob hashes and URLs, I/O contract, chaining
  measurements.json    every number, machine-readable
  README.md            the provenance ledger
  tokenizer.js         CLIP BPE, 12/12 against HuggingFace
  tokenizer/           vocab.json, merges.txt, tokenizer_meta.json
  verification/        reference case, bars and tolerances
scripts/
  verify.mjs           Playwright, real Chrome, real Core ML
  serve.mjs            static server with Range support
  publish-weights.sh   upload the blobs, write their URLs into the manifest
demo/index.html        prompt box, image, per-stage ms
```

## Adding an entry

1. **Record the graph.** Wrap the `MLGraphBuilder` you already build with in a
   Proxy that logs every call, and stream the constants out as they are created
   rather than accumulating them. 1.6 GB held a third time in JS will lose the
   WebNN context. The reference recorder is `bench/webnn/ir-record.js` in the
   `webnn-workbench` repo.

2. **Write `models/<id>/`**: the recipes, a `manifest.json` with the blob sizes
   and sha256s and the I/O contract, a `measurements.json`, a `verification/`
   directory, and a `README.md`.

3. **Check op coverage** with `opCoverage(recipe)`. If the loader is missing a
   call form, add it to `POSITIONAL` / `VARIADIC` / `MULTI_OUTPUT` in
   `runtime/loader.js`, and to the table in `runtime/README.md`.

4. **Verify on hardware.** `scripts/verify.mjs` is written around one entry
   today; generalising it is the obvious next task. An entry without a passing
   verification run is not an entry.

5. **Add a row to `catalog.json`**, and one to the table above.

### Four conventions

**A recipe is the exact call sequence, with backend-inferred shapes.** Not a
re-derivation. If the recorder computes the shapes itself, the recipe can
disagree with the graph that was measured, and eventually will.

**Constants never enter git.** A separate blob, pinned by sha256, whose byte
offsets the recipe indexes.

**A number without a machine, an OS, a browser build and a protocol is not a
measurement.** Every `measurements.json` carries all four, including how the A/B
was run and what the noise floor is.

**Every entry's README is a ledger.** What was folded and what it was worth;
what was tried and rejected, with the number that killed it; and what the
backend turned out to be like. The op list survives on its own. The reasoning
does not.
