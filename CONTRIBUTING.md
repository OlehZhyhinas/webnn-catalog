# Contributing

The [README](README.md) covers using the catalog. This file covers adding to it,
whether that's a timing row from your machine, an entry for new hardware, or a
whole new family.

There are two kinds of entry, distinguished by `runtimeKind`. A **WebNN** entry
has a recipe (the exact `MLGraphBuilder` call sequence) and a constants blob. A
**WebLLM** entry has hash-pinned runtime JavaScript and model-library WASM, and
points at an upstream model repository at a pinned revision.

## 1. A measurement row

This is the cheapest contribution and one of the more useful ones. Every entry so
far has been measured on a single machine, so its timings are one data point.
Running the same entry on different hardware gives it a range.

A second machine running an existing entry is a **row**, not a new entry:

```bash
node scripts/probe-target.mjs --out target.json
node scripts/add-measurement.mjs --entry <family>/<entry-id> \
  --results <results>.json --target target.json --protocol e2e
```

Each row has its own host fingerprint, so an entry can collect timings from
several machines without turning into several entries.

Some rules for rows. A number needs a machine, an OS, a browser build, and a
protocol attached to it, or it doesn't count as a measurement. Don't copy a load
average from a different run. If a number is a projection, label it as one and
keep it separate from measured values. And don't name whatever else was running
on your machine at the time. If load matters, describe it generically.

## 2. An entry

### WebNN

You need a tuning run that recorded its `MLGraphBuilder` call sequence, and the
constants blob that sequence indexes into.

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

# 3. Check it, against the schemas and against the hardware.
node scripts/validate.mjs
node scripts/verify.mjs --entry sd-turbo-512-1step/<entry-id>
```

`add-entry.mjs` reads the recipes to work out the entry's I/O, hashes the
constants blobs in place (they're gigabytes and don't go in git), writes
`entry.json`, `manifest.json`, and `measurements.json`, copies in the
verification set and the ledger, adds a summary row to `catalog.json`, and
refuses to overwrite an existing entry unless you pass `--force`.

### WebLLM

A WebLLM entry publishes hash-pinned artifacts and references upstream weights
at a fixed revision. It doesn't rehost the model.

```bash
node scripts/verify-artifacts.mjs --entry <family>/<entry-id>
WEBNN_MAX_LOAD=0 npm run verify:webllm
```

`verify-artifacts.mjs` checks each declared byte count and SHA-256.
`verify:webllm` runs the family's greedy corpus and requires the output to be
byte-identical to the baseline named in the entry. `scripts/gpu-lock.mjs`
serializes runs so two benchmarks don't end up sharing the GPU.

Each WebLLM entry needs a `NOTICE.md` with third-party attribution, and a
`source.license` field with an SPDX identifier tied to the upstream revision.
Don't add an entry without both.

### Every entry needs a ledger

`provenance.md` is required. It records what was folded and how many
milliseconds each fold saved, what was tried and dropped along with the number
that made you drop it, and anything you learned about how the backend behaves.
The op list on its own doesn't tell anyone why it looks the way it does.

### Publishing constants

```bash
node scripts/chunk-manifest.mjs --entry <family>/<entry-id>
scripts/publish-weights.sh <family>/<entry-id>
```

`chunk-manifest.mjs` splits a blob into Range-sized chunks, aiming for 64 MiB
each, with every boundary on a constant start. If several recipes read the same
blob, boundaries are placed where they're a constant start in all of them, so
nothing gets split across chunks and the loader can use the fetched bytes without
copying. The host you publish to needs to support Range requests and allow CORS.

## 3. A family

`family.json` is written by hand. It's the contract that every entry has to
match, so there's no tool that generates it.

A family is one model, one task, and one I/O contract, and every entry in it
meets that contract exactly. **If a change would break any part of the contract,
it's a new family, not a new entry.** A different resolution, step count,
tokenizer, or output shape all count as breaking it.

## Checks

```bash
npm install
node scripts/validate.mjs          # schema + cross-file checks
node scripts/check-chunks.mjs      # replay graphs off the chunk lists
npm test                           # loader unit tests
node scripts/verify.mjs --entry <family>/<entry-id> --weights <dir>
node scripts/serve.mjs             # then open demo/index.html or demo/qwen.html
```

`validate.mjs` has no dependencies. It implements the subset of JSON Schema the
schemas actually use, and it errors on any keyword it doesn't recognize instead
of ignoring it. After the schema pass it checks things a schema can't express:
that referenced files exist, that each recipe still hashes to what its manifest
says, that an entry's declared I/O matches its recipes and its family's
contract, that every chain link typechecks, and that the index agrees with the
entries it points at.

## Known gaps

**The ToDo fast variant of SD-Turbo has no entry.** Token Downsampling runs at
54.6 ms, about 20% faster, and produces a visibly different image (PSNR 24.3 dB
against the exact pipeline). It was measured end to end, but the
`MLGraphBuilder` call sequence was never recorded, so there's no recipe to
publish. The numbers are in the SD-Turbo entry's
[`provenance.md`](families/sd-turbo-512-1step/entries/coreml-apple-m5-pro-macos26-chrome152/provenance.md#the-optional-fast-variant).

**Every entry targets Apple silicon.** A DirectML or TFLite WebNN entry, or a
WebLLM entry on a non-Apple GPU, would be the most useful thing anyone could add
right now. The family/entry split exists for exactly that case, and nothing
exercises it yet.

**Recipe schema v1 can't distinguish call forms.** The recorder flattens
positional non-operand arguments into the options bag, so the JSON alone can't
tell `softmax(x, 2)` from `softmax(x, {axis: 2})`. `x-callForms.positional`
makes it recoverable. A v2 should record the argument list positionally.
