# runtime

One file, `loader.js`, an ES module with no dependencies. It takes a recipe and
a blob of constants and gives back a built `MLGraph`.

It knows nothing about diffusion, UNets or text encoders. It knows the WebNN
builder surface and the recipe schema, and it knows the two things about this
backend that will otherwise cost you an afternoon: the incognito gate and the
fact that every declared graph output must be bound at dispatch.

```js
import {
  loadRecipe, constantSource, assertCoreMLFingerprint,
} from "./runtime/loader.js";

const ctx = await navigator.ml.createContext({ deviceType: "gpu" });
assertCoreMLFingerprint(ctx);                       // fail loudly, not slowly

const manifest = await (await fetch("models/sd-turbo-512-1step/manifest.json")).json();
const recipe   = await (await fetch("models/sd-turbo-512-1step/recipe.image.json")).json();

const image = await loadRecipe(
  recipe,
  constantSource(manifest.constants.image, { baseUrl: "/weights" }),
  ctx,
);
// -> { graph, inputs, outputs, label, layout, stats }
```

## API

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
constantSource(manifestEntry, { baseUrl })        // from a manifest record
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

## Recipe schema, version 1

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

Operand names live in one namespace: graph inputs keep their own names,
constants are `k<N>`, op results are `v<N>`. A multi-output op (`split`) carries
`outputs` and `outputShapes` alongside `output`/`outputShape`; every name in
`outputs` is defined by that op.

An op's `inputs` array is `[...positional operands, ...operand-valued options]`,
in that order, because an options bag is always the last argument. `conv2d`'s
bias therefore appears twice, once as `options.bias` and once at the end of
`inputs`, and the loader shears the operand-valued options off the tail by
count to recover the positional list.

### The one thing the schema does not say

The recorder flattens a call's positional **non-operand** arguments into the
same `options` bag as the real options dictionary. So the recipe alone cannot
distinguish `softmax(x, 2)` from `softmax(x, {axis: 2})`, or
`split(x, 3, {axis: 1})` from `split(x, {splits: 3, axis: 1})`. The loader
carries a `POSITIONAL` table, the recorder's table inverted, naming which
keys are really positional and in what order:

```js
reshape: ["newShape"],  expand: ["newShape"],  softmax: ["axis"],
cast: ["type"],         concat: ["axis"],      split: ["splits"],
tile: ["repetitions"],  pad: ["beginningPadding", "endingPadding"],
argMin: ["axis"],       argMax: ["axis"],
```

Plus `VARIADIC` (`concat` takes an array of operands, not one) and
`MULTI_OUTPUT` (`split` returns several).

**Keep this table in sync with whatever recorder produced the recipe.** A
version 2 of the schema should either record the argument list positionally or
name the call form explicitly, and then the table can go.

## Op coverage

Every op type in the two SD-Turbo recipes, replayed and verified:

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
