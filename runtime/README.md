# runtime

One file, `loader.js`, an ES module with no dependencies. Hand it a catalog
entry and it gives back built `MLGraph`s; hand it a single recipe and a blob of
constants and it gives back one.

It knows nothing about diffusion, UNets or text encoders. It knows the WebNN
builder surface and the recipe schema, and it knows the three things about this
backend that will otherwise cost you an afternoon: the incognito gate, the fact
that every declared graph output must be bound at dispatch, and that an
intermediate should never cross into JS.

It also knows nothing about *which* entry you should be running. Selection is
the consuming product's policy; see the README's "Selection is the consumer's
job". This module starts after that decision.

```js
import {
  loadEntry, createEntryTensors, assertCoreMLFingerprint,
} from "./runtime/loader.js";

const dir   = "/families/sd-turbo-512-1step/entries/coreml-apple-m5-pro-macos26-chrome152";
const entry = await (await fetch(`${dir}/entry.json`)).json();

const ctx = await navigator.ml.createContext({ deviceType: "gpu" });
assertCoreMLFingerprint(ctx);                       // fail loudly, not slowly

const rig = await loadEntry(entry, "/weights", ctx, { baseUrl: dir });
const t   = await createEntryTensors(ctx, rig);
```

## API

### `loadEntry(entry, constants, context, opts?) -> Promise<Rig>`

Builds every graph the entry declares.

| | |
|---|---|
| `entry` | the parsed `entry.json` |
| `constants` | a base URL for the blobs, a `(name, manifestRecord) => source` factory, a `{name: source}` map, or `null` to use each manifest record's own `url` |
| `context` | an `MLContext` |
| `opts.baseUrl` | where the entry's own files live; defaults to `entry.baseUrl` or `"."` |
| `opts.only` | build a subset of the graphs |
| `opts.manifest`, `opts.recipes` | pass already-fetched JSON instead of refetching |
| `opts.order` | build order; the default is largest constants first, which keeps peak resident bytes lower |
| `opts.onProgress` | as `loadRecipe`, plus `graph` |

Returns `{entry, manifest, graphs, chain, inputs, outputs, stats}`, where
`graphs[name]` is exactly what `loadRecipe` returns and `chain` is
`entry.graphs.chain` parsed into `{from: {graph, name}, to: {graph, name}}`.

### `createEntryTensors(context, rig, opts?) -> Promise<Tensors>`

Allocates every tensor the entry needs and returns
`{tensors, inputsFor(graph), outputsFor(graph), get(graph, name)}`, ready to
hand straight to `dispatch()`.

Each chain link gets **one** MLTensor, bound as the producer's output and as the
consumer's input, so the intermediate never crosses into JS. An output that
feeds a link is allocated `readable`, because `readTensor()` on it is the
cheapest completion fence WebNN offers. An output the entry marks
`readable: false` is allocated non-readable and still bound, because WebNN
requires every declared output to be bound at dispatch.

`opts.readable` / `opts.writable` take `"<graph>.<name>"` keys and override
both defaults.

### `loadRecipe(recipe, source, context, opts?) -> Promise<Loaded>`

Replays the recipe's op list into a `MLGraphBuilder` and builds it.

| | |
|---|---|
| `recipe` | the parsed recipe JSON |
| `source` | a constant source (below), or a plain `ArrayBuffer` |
| `context` | an `MLContext` |
| `opts.onProgress` | `({phase: "constants"\|"ops", ...}) => void` |
| `opts.builder` | an existing `MLGraphBuilder`, to emit two recipes into one graph |

Returns `{graph, inputs, outputs, label, layout, stats}`. `inputs` and
`outputs` are `{name: {dataType, shape}}`: exactly what `dispatch()` must bind.

### Constant sources

```js
bufferSource(arrayBuffer)                         // already in hand
httpSource(url, { chunks, totalBytes })           // one fetch, or ranged fetches
constantSource(manifestRecord, { baseUrl })       // from an entry manifest record
```

A source is `{ranges, totalBytes, fetchRange(offset, length)}`. Anything with
that shape works, so an OPFS cache or an IndexedDB store drops straight in.

**Whole file** is the default and the simplest: one fetch, one `ArrayBuffer`.
For the SD-Turbo image graph that is 1.65 GB in the JS heap at once, on top of
what Chrome copies into the graph.

**Chunked** is what a published, `Range`-served blob wants. Put a `chunks`
array in the manifest and each range is fetched only when its constants are
about to be built, then released:

```json
"chunks": [
  { "byteOffset": 0,         "byteLength": 201326592 },
  { "byteOffset": 201326592, "byteLength": 201326592 }
]
```

Peak resident bytes become one chunk instead of the whole blob. Chunk
boundaries do not have to respect constant boundaries: a constant that straddles
a seam is copied out and stitched when the next range arrives.

Constants are always created in `byteOffset` order, whatever order the ops use
them in. That is what makes releasing each chunk safe.

### `assertCoreMLFingerprint(context, {throwOnMismatch})`

Returns `{ok, preferredInputLayout, maxRank, backend}` and, by default, throws
with instructions if this is not the Core ML backend.

Chromium gates Core ML on `!is_incognito`. An off-the-record profile, which is
what `chromium.launch()` and an incognito window both give you, silently lands
on TFLite/XNNPACK on the CPU. Nothing errors. The graph builds in milliseconds
instead of ~25 seconds and runs about 50x slower. Two tells:

| | Core ML | TFLite/XNNPACK |
|---|---|---|
| `preferredInputLayout` | `nchw` | `nhwc` |
| `rankRange.max` | 5 | 8 |

Call it immediately after `createContext()`, before spending a minute building
a graph you are about to mis-measure.

### `opCoverage(recipe)` / `checkOpSupport(recipe, ctxOrBuilder)`

`opCoverage` returns a `{opType: count}` histogram. `checkOpSupport` returns
`{ok, missing}` after checking each type against the browser's
`MLGraphBuilder`. Both are cheap and neither touches the constants, so you can
find out a browser is missing an op before downloading 1.6 GB.

### Tensor helpers

`createInputTensors`, `createOutputTensors`, `chainTensor`, `writeTensor`,
`readTensor`, `fence`, `dispatch`. Thin, but they keep call sites symmetrical
and they encode the two rules below.

## The MLTensor chaining pattern

A two-graph pipeline should never move an intermediate through JS. The tensor
that receives graph A's output **is** the tensor bound to graph B's input:

```js
// ONE MLContext for both graphs.
const embedding = await ctx.createTensor({
  dataType: "float16", shape: [1, 77, 1024], readable: true,
});

ctx.dispatch(text.graph,  { input_ids },        { out: embedding });
await ctx.readTensor(embedding);   // a completion FENCE, not a transfer

ctx.dispatch(image.graph, { sample, encoder_hidden_states: embedding, latent_raw },
                          outs);
await ctx.readTensor(outs.out, hostView);   // 1 MB, into a buffer you own
```

Three things are load-bearing here.

**`readable: true` on the intermediate.** Not because the value is read, but
because `readTensor` on it is the cheapest completion fence WebNN offers.
`dispatch()` is queued; reading any output of the last queued graph resolves
only once it has run. On this pipeline that fence costs about 0.1 ms.

**The value never leaves the device.** The `[1,77,1024]` float16 embedding is
157 KB that would otherwise cross into JS and back. If the two graphs are on
*different* `MLContext`s, it must: read it back, write it across. That costs
about 0.1 ms, and on the ANE 2.7x the float16 error, which is why the
text-encoder-on-npu experiment was rejected.

**Re-running the same prompt skips the text graph entirely.** The embedding
tensor still holds it. That is the difference between the 67.85 ms new-prompt
and ~60 ms cached numbers.

## Two rules the API will not let you forget

**Every declared graph output must be bound at dispatch.** A recipe recorded
with a debug output still declares it. Allocate it non-readable and bind it;
`createOutputTensors(ctx, loaded, {readable: ["out"]})` does exactly that.

**Read back into a buffer you own.** `readTensor(tensor, view)`, the two-
argument overload in Chrome 152, fills a view allocated once rather than
handing back a fresh `ArrayBuffer` per frame. With an RGBA-packed int32 output
you can then keep a permanent `ImageData` over that buffer, and "convert to
pixels" becomes literally nothing.

## Recipe schema

The recipe IR is defined by [`../schema/recipe.schema.json`](../schema/recipe.schema.json)
and explained in the top-level README under "The recipe IR": the single operand
namespace, the operand-valued options that appear twice, the `x-callForms`
positional table that tells `softmax(x, 2)` from `softmax(x, {axis: 2})`, and
the monotonic constant offsets that make chunked loading safe.

The loader's `POSITIONAL`, `VARIADIC` and `MULTI_OUTPUT` tables are that
schema's `x-callForms` in code. **Keep them in sync with whatever recorder
produced the recipe**, and with the schema, which is the authority.

## Op coverage

Every op type in the SD-Turbo entry's two recipes, replayed and verified:

| | image | text |
|---|---:|---:|
| `add` | 254 | 162 |
| `cast` | 2 |  |
| `clamp` | 1 |  |
| `concat` | 21 |  |
| `conv2d` | 111 |  |
| `gather` |  | 1 |
| `gelu` | 16 | 23 |
| `layerNormalization` | 109 | 47 |
| `matmul` | 224 | 138 |
| `mul` | 125 |  |
| `relu` | 31 |  |
| `resample2d` | 3 |  |
| `reshape` | 305 | 94 |
| `roundEven` | 1 |  |
| `sigmoid` | 45 |  |
| `softmax` | 32 | 23 |
| `split` | 32 | 23 |
| `tanh` | 1 |  |
| `transpose` | 128 | 92 |
| **total** | **1441** | **603** |

The replay itself costs about 8 ms for 1441 ops. The 25 s is Core ML compiling
the result, and Chromium caches none of it.
